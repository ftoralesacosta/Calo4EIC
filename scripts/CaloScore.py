import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
# import horovod.tensorflow.keras as hvd
import utils
from architectures import ConvModel, Unet, Resnet
import time
# tf and friends
#tf.random.set_seed(1234)

class CaloScore(keras.Model):
    """Score based generative model"""
    def __init__(self, num_layer,num_cond=1,name='SGM',config=None):
        super(CaloScore, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.num_cond = num_cond        
        self.config = config
        self.num_embed = self.config['EMBED']
        self.data_shape = self.config['SHAPE'][1:]
        self.num_layer = num_layer
        self.num_steps = 512
        self.ema=0.999
                

        self.timesteps =tf.range(start=0,limit=self.num_steps + 1, dtype=tf.float32) / self.num_steps + 8e-3 
        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        self.betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(alphas, 0)
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - self.alphas_cumprod)
        
                
        # self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        #Convolutional model for 3D images and dense for flatten inputs
            
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.activation = tf.keras.activations.swish

        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_layer = Input((self.num_layer))

        voxel_conditional = self.Embedding(inputs_time,self.projection)
        layer_conditional = self.Embedding(inputs_time,self.projection)

        voxel_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [voxel_conditional,inputs_layer,inputs_cond],-1))
        
        layer_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [layer_conditional,inputs_cond],-1))
        
        
        self.shape = (-1,1,1,1,1)

        inputs,outputs = Unet(
            self.data_shape,
            voxel_conditional,
            input_embedding_dims = 32,
            stride=2,
            kernel=3,
            block_depth = 2,
            widths = [32,64,96],
            attentions = [False,False, True],
            pad=config['PAD'],
        )

        
        self.model_voxel = keras.Model(inputs=[inputs,inputs_time,inputs_layer,inputs_cond],
                                       outputs=outputs)

        outputs = Resnet(
            inputs_layer,
            self.num_layer,
            layer_conditional,
            num_embed=self.num_embed,
            num_layer = 5,
            mlp_dim= 512,
        )
        
        self.model_layer = keras.Model(inputs=[inputs_layer,inputs_time,inputs_cond],
                                       outputs=outputs)

        self.ema_layer = keras.models.clone_model(self.model_layer)
        self.ema_voxel = keras.models.clone_model(self.model_voxel)

        if self.verbose:
            print(self.model_voxel.summary())


        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        half_dim = self.num_embed // 4
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb * tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding
        
        
    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions)

    @tf.function
    def train_step(self, inputs):
        voxel,layer,cond = inputs

        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        #random_t = tf.cast(random_t,tf.float32)
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        sigma = tf.clip_by_value(sigma, clip_value_min = 1e-3, clip_value_max=0.999)

        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            
        with tf.GradientTape() as tape:
            #voxel
            z = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
            perturbed_x = alpha_reshape*voxel + z * sigma_reshape
            score = self.model_voxel([perturbed_x, random_t,layer,cond])            
            v = alpha_reshape * z - sigma_reshape * voxel
            losses = tf.square(score - v)            
            loss_voxel = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)), axis=-1)

        trainable_variables = self.model_voxel.trainable_variables
        g = tape.gradient(loss_voxel, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_voxel.weights, self.ema_voxel.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        with tf.GradientTape() as tape:
            #layer
            z = tf.random.normal((tf.shape(layer)))
            perturbed_x = alpha*layer + z * sigma            
            score = self.model_layer([perturbed_x, random_t,cond])
            v = alpha * z - sigma * layer
            losses = tf.square(score - v)
                        
            loss_layer = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)), axis=-1)
            
        trainable_variables = self.model_layer.trainable_variables
        g = tape.gradient(loss_layer, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_layer.weights, self.ema_layer.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        self.loss_tracker.update_state(loss_voxel + loss_layer)
        
        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":tf.reduce_mean(loss_voxel),
            "loss_layer":tf.reduce_mean(loss_layer),
        }

    @tf.function
    def test_step(self, inputs):
        voxel,layer,cond = inputs
        
        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
        
        #random_t = tf.cast(random_t,tf.float32)
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        sigma = tf.clip_by_value(sigma, clip_value_min = 1e-3, clip_value_max=0.999)
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
        
        
        #voxel
        z = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
        perturbed_x = alpha_reshape*voxel + z * sigma_reshape
        


        score = self.model_voxel([perturbed_x, random_t,layer,cond])
        denoise = alpha_reshape * z - sigma_reshape * voxel
        losses = tf.square(score - denoise)
        
        loss_voxel = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)), axis=-1)
        
        #layer
        z = tf.random.normal((tf.shape(layer)))
        perturbed_x = alpha*layer + z * sigma            
        score = self.model_layer([perturbed_x, random_t,cond])
        denoise = alpha_reshape * z - sigma * layer
        losses = tf.square(score - denoise)
        
        loss_layer = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)), axis=-1)
        loss = tf.reduce_mean(loss_voxel+loss_layer)

        
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":tf.reduce_mean(loss_voxel),
            "loss_layer":tf.reduce_mean(loss_layer),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def generate(self,cond):
        start = time.time()
        layer_energy = self.DDPMSampler(cond,self.ema_layer,
                                        data_shape=[self.num_layer],
                                        const_shape = [-1,1])

        voxels = self.DDPMSampler(cond,self.ema_voxel,
                                  data_shape = self.data_shape,
                                  const_shape = self.shape,
                                  layer_energy=layer_energy)
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return voxels.numpy(),layer_energy.numpy()
        

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    layer_energy=None):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0]
        t = tf.ones((batch_size,1))
        data_shape = np.concatenate(([batch_size],data_shape))
        init_x = self.prior_sde(data_shape)
        
        x = init_x
        

        for time_step in tf.range(self.num_steps, -1, delta=-1):
            batch_time_step = tf.ones((batch_size,1),dtype=tf.int32) * time_step
            z = tf.random.normal(x.shape)

            alpha = tf.gather(tf.sqrt(self.alphas_cumprod),batch_time_step)
            sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),batch_time_step)
            
            if layer_energy is None:
                score = model([x, batch_time_step,cond],training=False)
            else:
                score = model([x, batch_time_step,layer_energy,cond],training=False)
                alpha = tf.reshape(alpha,self.shape)
                sigma = tf.reshape(sigma,self.shape)

                
            x_recon = alpha * x - sigma * score

            p1 = tf.reshape(tf.gather(self.posterior_mean_coef1,batch_time_step),const_shape)
            p2 = tf.reshape(tf.gather(self.posterior_mean_coef2,batch_time_step),const_shape)
            mean = p1*x_recon + p2*x
           
            log_var = tf.reshape(tf.gather(tf.math.log(self.posterior_variance),batch_time_step),const_shape)

            x = mean + tf.exp(0.5 * log_var) * z
            
        

        # The last step does not include any noise
        return mean

        
