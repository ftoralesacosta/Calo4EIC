import json, yaml
import os
import h5py as h5
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick

def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    # print(tf.data.experimental.cardinality(test_data).numpy(),"cardinality")
    # input()
    return train_data,test_data

line_style = {
    'Geant4':'dotted',
    'CaloScore':'-',
    'FPCD':'--',
    # 'WGAN-GP':'-',
}

colors = {
    'Geant4':'black',
    'CaloScore':'#7570b3',
    'FPCD':'darkorange',
    # 'WGAN-GP':'#e7298a',
}

name_translate={
    'CaloScore':'CaloScore',
    'FPCD':'FPCD',
    # 'wgan':'WGAN-GP',
}

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs


def GetEMD(ref,array):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(ref,array)
    # mse = np.square(ref-array)/ref
    # return np.sum(mse)

def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',emd=True):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if emd==False or reference_name==plot:
            plot_label = plot
        else:
            emdval = GetEMD(np.mean(feed_dict[reference_name],0),np.mean(feed_dict[plot],0))
            plot_label = r"{}, EMD :{:.2f}".format(plot,emdval)
            
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot_label,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot_label,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=13,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best',emd=True):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),30)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    
    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            dist,_ = np.histogram(feed_dict[plot],bins=binning,density=True)
            ax0.plot(xaxis,dist,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3,label=plot)
            #dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,marker=line_style[plot],color=colors[plot],density=True,histtype="step")
        else:
            dist,_ = np.histogram(feed_dict[plot],bins=binning,density=True)
            if emd==False or reference_name==plot:
                plot_label = plot
            else:
                emdval = GetEMD(feed_dict[reference_name],feed_dict[plot])
                plot_label = r"{}, EMD :{:.2f}".format(plot,emdval)
                
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot_label,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
            
        if reference_name!=plot:
            ratio = 100*np.divide(reference_hist-dist,reference_hist)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(xaxis,ratio,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3)
            else:
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=13,ncol=1)        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 

    if logy:
        ax0.set_yscale('log')
    
    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0



def EnergyLoader(file_name,nevts,emax,emin,logE=True):
    
    with h5.File(file_name,"r") as h5f:
        e = h5f['cluster'][-int(nevts):,:1].astype(np.float32)
    if logE:        
        return np.log10(e/emin)/np.log10(emax/emin)
    else:
        return (e-emin)/(emax-emin)
        



def DataLoader(file_name,shape,
               nevts,emax,emin,
               max_deposit=2,
               logE=True,
               use_1D=False,
               rank=0,size=1):
    

    with h5.File(file_name,"r") as h5f:
        e = h5f['cluster'][rank:int(nevts):size,:1].astype(np.float32)
        shower = h5f['calo_images'][rank:int(nevts):size].astype(np.float32)

        
    shower = shower.reshape(shape)
    layer = np.sum(shower,(2,3,4),keepdims=True)
    shower = np.ma.divide(shower,layer)
    def convert_energies(e,layer_energies):
        converted = np.zeros(layer_energies.shape,dtype=np.float32)
        converted[:,0] = np.ma.divide(np.sum(layer_energies,-1),np.squeeze(max_deposit*e)).filled(0)
        #print(np.ma.divide(np.sum(layer_energies,-1),np.squeeze(max_deposit*e)).filled(0))
        #input()
        for i in range(1,layer_energies.shape[1]):
            converted[:,i] = np.ma.divide(layer_energies[:,i-1],np.sum(layer_energies[:,i-1:],-1)).filled(0)
            
        return converted

    layer = convert_energies(e,np.squeeze(layer))

    if logE:        
        return shower,layer,np.log10(e/emin)/np.log10(emax/emin)
    else:
        return shower,layer,(e-emin)/(emax-emin)
        
def ReverseNorm(voxels,layer,e,
                emax,emin,max_deposit,logE=True,
                datasetN=2):
    '''Revert the transformations applied to the training set'''
    #shape=voxels.shape

    if logE:
        energy = emin*(emax/emin)**e
    else:
        energy = emin + (emax-emin)*e

    def _revert(x,fname='jet'):
        params = LoadJson(fname)
        x = x*params['std'] + params['mean']
        x = revert_logit(x)        
        x = x * (np.array(params['max'])-params['min']) + params['min']
        return x

    voxels = _revert(voxels,'preprocessing_{}_voxel.json'.format(datasetN))
    layer = _revert(layer,'preprocessing_{}_layer.json'.format(datasetN))

    #Undo layer energy transformation
    layer_norm= np.zeros(layer.shape,dtype=np.float32)    
    for i in range(layer.shape[1]):
        layer_norm[:,i] = np.squeeze(max_deposit*energy)*layer[:,i]
        
    layer_norm[:,0] = np.squeeze(max_deposit*energy)*layer[:,0]*layer[:,1]
    for i in range(1,layer.shape[1]-1):
        layer_norm[:,i] = layer[:,i+1]*(np.squeeze(max_deposit*energy)*layer[:,0] - np.sum(layer_norm[:,:i],-1))
    layer_norm[:,-1] = np.squeeze(max_deposit*energy)*layer[:,0] - np.sum(layer_norm[:,:-1],-1)


    voxels /= np.sum(voxels,(2,3,4),keepdims=True)
    voxels*=layer_norm.reshape((-1,layer_norm.shape[1],1,1,1))
        
    
    return voxels,energy



def logit(x):             
    alpha = 1e-6
    x = alpha + (1 - 2*alpha)*x
    return np.ma.log(x/(1-x)).filled(0)
    
def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

    
def CalcPreprocessing(data,fname):
    '''Apply data preprocessing'''
    
    data_dict = {
        'max':np.max(data,0).tolist(),
        'min':np.min(data,0).tolist(),
    }

    data = np.ma.divide(data-data_dict['min'],np.array(data_dict['max'])- data_dict['min']).filled(0)
    data = logit(data)
        
    mean = np.average(data,axis=0)
    std = np.std(data,axis=0)
    data_dict['mean']=mean.tolist()
    data_dict['std']=std.tolist()    
    SaveJson(fname,data_dict)

    data = (np.ma.divide((data-data_dict['mean']),data_dict['std']).filled(0)).astype(np.float32)
    
    print("done!")
    return data


def ApplyPreprocessing(data,fname):
    data_dict = LoadJson(fname)
    data = np.ma.divide(data-data_dict['min'],np.array(data_dict['max'])- data_dict['min']).filled(0)
    data = logit(data)
    data = (np.ma.divide((data-data_dict['mean']),data_dict['std']).filled(0)).astype(np.float32)
    return data
    

def SaveJson(save_file,data,base_folder=''):
    with open(os.path.join(base_folder,save_file),'w') as f:
        json.dump(data, f)

    
def LoadJson(file_name,base_folder=''):
    import json,yaml
    JSONPATH = os.path.join(base_folder,file_name)
    return yaml.safe_load(open(JSONPATH))

