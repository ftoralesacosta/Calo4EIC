import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
# import horovod.tensorflow.keras as hvd
from CaloScore import CaloScore
import gc
from CaloScore_distill import CaloScore_distill
from sklearn.metrics import roc_curve, auc
from WGAN import WGAN
import time
# hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# rank = hvd.rank()
# size = hvd.size()
# tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
utils.SetStyle()

nevents_hard = 100_000
nevents_hard = 10_000

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='./', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
parser.add_argument('--config', default='config.json', help='Training parameters')
parser.add_argument('--nevts', type=float,default=1e5, help='Number of events to load')

#Parallel generation parameters
parser.add_argument('--model', default='CaloScore', help='Type of generative model to load')
parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
parser.add_argument('--test', action='store_true', default=False,help='Verify the reverse transform is correct')
parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')


parser.add_argument('--sample', action='store_true', default=False,help='Sample from learned model')

flags = parser.parse_args()

nevts = int(flags.nevts)
config = utils.LoadJson(flags.config)
emax = config['EMAX']
emin = config['EMIN']
run_classifier=False

if flags.sample:
    checkpoint_folder = '../checkpoints_{}_{}'.format(config['CHECKPOINT_NAME'],flags.model)
    energies = []
    for dataset in config['EVAL']:
        e_ = utils.EnergyLoader(os.path.join(flags.data_folder,dataset),
                                flags.nevts,
                                emax = config['EMAX'],
                                emin = config['EMIN'],
                                logE=config['logE'])
        energies.append(e_)

    energies = np.reshape(energies, (-1, 1))
    # print(energies)

    if flags.model == 'wgan':
        num_noise = config['NOISE_DIM']
        model = WGAN(config['SHAPE_PAD'][1:],energies.shape[1],config=config,num_noise=num_noise)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
        start = time.time()
        generated = model.generate(energies.shape[0], energies)
        end = time.time()
        print(end - start)
    else:
        model = CaloScore(num_layer=config['NUM_LAYER'], config=config)
        if flags.distill:
            checkpoint_folder = '../checkpoints_{}_{}_d{}'.format(config['CHECKPOINT_NAME'],flags.model,flags.factor)

            model = CaloScore_distill(model.ema_layer,model.ema_voxel,
                                      factor=flags.factor,
                                      num_layer = config['NUM_LAYER'],
                                      config=config,)
            print("Loading distilled model from: {}".format(checkpoint_folder))
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()

        nsplit = 50
        generated = []
        layers = []
        for i,split in enumerate(np.array_split(energies,nsplit)):
            #if i> 5: break
            v,l = model.generate(cond=np.squeeze(split))
            generated.append(v)
            layers.append(l)            
            gc.collect()
            
        generated = np.concatenate(generated)
        layers = np.concatenate(layers)

        # generated,layers = model.generate(cond=np.squeeze(energies))
        generated,energies = utils.ReverseNorm(
            generated,layers,energies[:layers.shape[0]],
            logE=config['logE'],
            max_deposit=config['MAXDEP'],
            emax = config['EMAX'],
            emin = config['EMIN'],
            datasetN=config['DATASET'],
        )
        
    generated[generated<config['ECUT']] = 0 #min from samples

    print('generated_{}_{}.h5'.format(config['CHECKPOINT_NAME'],flags.model))
    with h5.File(os.path.join(flags.data_folder,'CaloScore_images_5x5.h5'.format(config['CHECKPOINT_NAME'],flags.model,flags.factor)),"w") as h5f:
        dset = h5f.create_dataset("calo_images", data=np.reshape(generated,(generated.shape[0],-1)))
        dset = h5f.create_dataset("cluster", data=energies)
else:

    def LoadSamples(model):
        generated = []
        energies = []
        # with h5.File(os.path.join(flags.data_folder,'CaloScore_images_5x5.h5'.format(config['CHECKPOINT_NAME'],flags.model,flags.factor)),"r") as h5f:
        with h5.File(os.path.join(flags.data_folder,f'{model}_images_5x5.h5'),"r") as h5f:
        # with h5.File(os.path.join(flags.data_folder,f'{model}_images_1x1.h5'),"r") as h5f:
            if (model == 'FPCD'):
                energies.append(h5f['truth_features'][-nevents_hard:, :1])
                generated.append(h5f['calo_images'][-nevents_hard:])
            else:
                energies.append(h5f['cluster'][:,:1])
                generated.append(h5f['calo_images'][:])
        print("ENERGIES FOR MODEL", model, energies[-10:])
        energies = np.reshape(energies,(-1,1))
        print("Loaded {} Samples".format(energies.shape[0]))
        generated = np.reshape(generated,config['SHAPE'])
        return generated,energies

    def LoadTest(nevts=-1):
        for dataset in config['EVAL']:    
            voxel_,layer_,energy_, = utils.DataLoader(
                os.path.join(flags.data_folder,dataset),
                config['SHAPE'],nevts,
                emax = config['EMAX'],emin = config['EMIN'],
                max_deposit=config['MAXDEP'], #noise can generate more deposited energy than generated
                logE=config['logE'],
                # rank=hvd.rank(),size=hvd.size(),
                use_1D = config['DATASET']==1,
            )
            voxel_ = utils.ApplyPreprocessing(
                voxel_,
                "preprocessing_{}_voxel.json".format(config['DATASET']))
            layer_ = utils.ApplyPreprocessing(
                layer_,
                "preprocessing_{}_layer.json".format(config['DATASET']))
            
            generated,energies = utils.ReverseNorm(
                voxel_,layer_,energy_,
                logE=config['logE'],
                max_deposit=config['MAXDEP'],
                emax = config['EMAX'],
                emin = config['EMIN'],
                datasetN=config['DATASET'],
            )
            energies = np.reshape(energies,(-1,1))
            generated = np.reshape(generated,config['SHAPE'])
            return generated, energies

    if flags.model != 'all':
        # models = ['FPCD']
        models = ['CaloScore', 'FPCD']
    else:
        # models = ['VPSDE','subVPSDE','VESDE','wgan','vae']
        # models = ['CaloScore', 'wgan']
        models = ['CaloScore', 'FPCD']

    print("%"*30)
    print(models)
    if flags.test:
        data, energies = LoadTest(flags.nevts)
        data_dict = {
            'CaloScore':data
        }
        data_dict['CaloScore'][data_dict['CaloScore']<config['ECUT']] = 0 #min from samples

    else:
        energies = []
        data_dict = {}
        for model in models:
            print("\n\n MODEL = ", model, "\n\n")
            if np.size(energies) == 0:
                data, energies = LoadSamples(model)
                data_dict[utils.name_translate[model]] = data
                # print(data_dict[utils.name_translate[model]])
            else:
                data_dict[utils.name_translate[model]]=LoadSamples(model)[0]

    total_evts = energies.shape[0]


    data = []
    true_energies = []
    for dataset in config['EVAL']:
        with h5.File(os.path.join(flags.data_folder,dataset),"r") as h5f:
            true_energies.append(h5f['cluster'][-total_evts:,:1])
            data.append(h5f['calo_images'][-total_evts:])

    
    data_dict['Geant4']=np.reshape(data,config['SHAPE'])
    data_dict['Geant4'][data_dict['Geant4']<config['ECUT']] = 0 #min from samples
    true_energies = np.reshape(true_energies,(-1,1))

    
    #Plot high level distributions and compare with real values
    print("%"*40)
    print(true_energies)
    print(energies)
    assert np.allclose(true_energies,energies), 'ERROR: Energies between samples dont match'


    def ScatterESplit(data_dict,true_energies):
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        fig,ax = SetFig("Gen. energy [GeV]","Dep. energy [GeV]")
        
        for key in data_dict:
            #print(np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1).shape,true_energies.flatten().shape)
            ax.scatter(
                true_energies.flatten()[:500],
                np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)[:500],
                label=key)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc='best',fontsize=16,ncol=1)
        fig.savefig('{}/FCC_Scatter_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))


    def AverageShowerWidth(data_dict):
        eta_bins = config['SHAPE'][2]
        eta_binning = np.linspace(-1,1,eta_bins+1)
        eta_coord = [(eta_binning[i] + eta_binning[i+1])/2.0 for i in range(len(eta_binning)-1)]

        def GetMatrix(sizex,sizey,minval=-1,maxval=1):
            nbins = sizex
            binning = np.linspace(minval,maxval,nbins+1)
            coord = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
            matrix = np.repeat(np.expand_dims(coord,-1),sizey,-1)
            return matrix

        
        eta_matrix = GetMatrix(config['SHAPE'][2],config['SHAPE'][3])
        eta_matrix = np.reshape(eta_matrix,(1,1,eta_matrix.shape[0],eta_matrix.shape[1],1))
        
        
        phi_matrix = np.transpose(GetMatrix(config['SHAPE'][3],config['SHAPE'][2]))
        phi_matrix = np.reshape(phi_matrix,(1,1,phi_matrix.shape[0],phi_matrix.shape[1],1))

        def GetCenter(matrix,energies,power=1):
            ec = energies*np.power(matrix,power)
            sum_energies = np.sum(np.reshape(energies,(energies.shape[0],energies.shape[1],-1)),-1)
            ec = np.reshape(ec,(ec.shape[0],ec.shape[1],-1)) #get value per layer
            ec = np.ma.divide(np.sum(ec,-1),sum_energies).filled(0)

            return ec

        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width

        
        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_eta = {}
        feed_dict_eta2 = {}
        
        for key in data_dict:
            feed_dict_phi[key] = GetCenter(phi_matrix,data_dict[key])
            feed_dict_phi2[key] = GetWidth(feed_dict_phi[key],GetCenter(phi_matrix,data_dict[key],2))
            feed_dict_eta[key] = GetCenter(eta_matrix,data_dict[key])
            feed_dict_eta2[key] = GetWidth(feed_dict_eta[key],GetCenter(eta_matrix,data_dict[key],2))
            

        fig,ax0 = utils.PlotRoutine(feed_dict_eta,xlabel='Layer number', ylabel= 'x-center of energy')
        fig.savefig('{}/FCC_EtaEC_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi,xlabel='Layer number', ylabel= 'y-center of energy')
        fig.savefig('{}/FCC_PhiEC_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_eta2,xlabel='Layer number', ylabel= 'x-width')
        fig.savefig('{}/FCC_EtaW_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi2,xlabel='Layer number', ylabel= 'y-width')
        fig.savefig('{}/FCC_PhiW_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))

        return feed_dict_eta2

    def AverageELayer(data_dict):
        
        def _preprocess(data):
            print(">"*10,data.shape,total_evts,config['SHAPE'][1],-1)
            # preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(data,(data.shape[0],
                                                    config['SHAPE'][1],-1))
            # preprocessed = np.reshape(data,(total_evts,config['SHAPE'][1],-1))
            preprocessed = np.sum(preprocessed,-1)
            #preprocessed = np.mean(preprocessed,0)
            return preprocessed
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean deposited energy [GeV]')
        fig.savefig('{}/FCC_EnergyZ_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def AverageEX(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],config['SHAPE'][3],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed
            
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='x-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyX_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict
        
    def AverageEY(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,2,1,3,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],config['SHAPE'][2],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='y-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyY_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def HistEtot(data_dict):
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed,-1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

            
        binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),30)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
        ax0.set_xscale("log")
        fig.savefig('{}/FCC_TotalE_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict
        
    def HistNhits(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0], -1))
            return np.sum(preprocessed>0,-1)
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits', ylabel= 'Normalized entries',label_loc='upper left')
        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        fig.savefig('{}/FCC_Nhits_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict


    # def Classifier(data_dict,gen_name='CaloScore'):
    def Classifier(data_dict,gen_name='FPCD'):
        from tensorflow import keras
        train = np.concatenate([data_dict['Geant4'],data_dict[gen_name]],0)
        labels = np.concatenate([np.zeros((data_dict['Geant4'].shape[0],1)),
                                 np.ones((data_dict[gen_name].shape[0],1))],0)
        train=train.reshape((train.shape[0],-1))
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1,activation='sigmoid')
        ])
        opt = tf.optimizers.Adam(learning_rate=2e-4)
        model.compile(optimizer=opt,
                      loss="binary_crossentropy",
                      metrics=['accuracy'])
        
        model.fit(train, labels,batch_size=100, epochs=30)
        pred = model.predict(train)
        fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)    
        print("{} AUC: {}".format(auc(fpr, tpr),gen_name))


    
    def HistMaxELayer(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],config['SHAPE'][1],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Max. voxel/Dep. energy')
        fig.savefig('{}/FCC_MaxEnergyZ_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def HistMaxE(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0,1,10)
        fig,ax0 = utils.HistRoutine(feed_dict,ylabel='Normalized entries', xlabel= 'Max. voxel/Dep. energy',binning=binning,logy=True)
        fig.savefig('{}/FCC_MaxEnergy_{}_{}.pdf'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model))
        return feed_dict

    def Plot_Shower_2D(data_dict):
        #cmap = plt.get_cmap('PiYG')
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad("white")
        plt.rcParams['pcolor.shading'] ='nearest'
        layer_number = [1,5,10]
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        for layer in layer_number:
            
            def _preprocess(data):
                preprocessed = data[:,layer,:]
                preprocessed = np.mean(preprocessed,0)
                preprocessed[preprocessed==0]=np.nan
                return preprocessed

            vmin=vmax=0
            for ik,key in enumerate(['Geant4',utils.name_translate[flags.model]]):
                fig,ax = SetFig("x-bin","y-bin")
                average = _preprocess(data_dict[key])
                if vmax==0:
                    vmax = np.nanmax(average[:,:,0])
                    vmin = np.nanmin(average[:,:,0])
                    print(vmin,vmax)
                im = ax.pcolormesh(range(average.shape[0]), range(average.shape[1]), average[:,:,0], cmap=cmap)

                yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
                yScalarFormatter.set_powerlimits((0,0))
                #cbar.ax.set_major_formatter(yScalarFormatter)

                cbar=fig.colorbar(im, ax=ax,label='Dep. energy [GeV]',format=yScalarFormatter)
                
                
                bar = ax.set_title("{}, layer number {}".format(key,layer),fontsize=15)
                # ax.set_xlim(1,130)

                fig.savefig('{}/FCC_{}2D_{}_{}_{}.pdf'.format(flags.plot_folder,key,layer,config['CHECKPOINT_NAME'],flags.model))
            

    high_level = []
    plot_routines = {
        'Energy per layer':AverageELayer,
        'Energy':HistEtot,
        '2D Energy scatter split':ScatterESplit,
        'Nhits':HistNhits,
    }
    
    if '1' in flags.config:
        pass
        plot_routines['Max voxel']=HistMaxE
    else:
        pass
        plot_routines['Shower width']=AverageShowerWidth        
        plot_routines['Energy per eta']=AverageEX
        plot_routines['Energy per phi']=AverageEY
        plot_routines['2D average shower']=Plot_Shower_2D
        plot_routines['Max voxel']=HistMaxELayer
        plot_routines['Class']=Classifier
        

        
    for plot in plot_routines:
        if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
        print(plot)
        if 'split' in plot:
            plot_routines[plot](data_dict,true_energies)
        else:
            high_level.append(plot_routines[plot](data_dict))
            
