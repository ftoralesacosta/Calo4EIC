import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import utils
import time
# tf and friends
tf.random.set_seed(1234)

class CaloScore_distill(keras.Model):
    """Score based generative model distilled"""
    def __init__(self, teacher_layer,teacher_voxel,factor,num_layer,config=None):
        super(CaloScore_distill, self).__init__()
        
        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.factor = factor
        self.data_shape = self.config['SHAPE'][1:]
        self.num_layer = num_layer
        self.ema=0.999
        self.num_steps = 512//self.factor
        self.verbose=False

        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
        else:
            self.shape = (-1,1,1,1,1)

        self.betas,self.alphas_cumprod,self.alphas = self.get_alpha_beta(self.num_steps)
        self.teacher_betas,self.teacher_alphas_cumprod,_ = self.get_alpha_beta(2*self.num_steps)
        
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        self.loss_tracker = keras.metrics.Mean(name="loss")


        self.teacher_layer = teacher_layer
        self.teacher_voxel = teacher_voxel
        
        self.model_layer = keras.models.clone_model(teacher_layer)
        self.model_voxel = keras.models.clone_model(teacher_voxel)
        self.ema_layer = keras.models.clone_model(self.model_layer)
        self.ema_voxel = keras.models.clone_model(self.model_voxel)

        if self.verbose:
            print(self.model_voxel.summary())
        self.teacher_layer.trainable = False    
        self.teacher_voxel.trainable = False    
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        if self.verbose:
            print(self.model_voxel.summary())
            
        self.factor = factor


        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def get_alpha_beta(self,num_steps):
        timesteps =tf.range(start=0,limit=num_steps + 1, dtype=tf.float32) / num_steps + 8e-3 
        alphas = timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - betas
        alphas_cumprod = tf.math.cumprod(alphas, 0)
        return betas,alphas_cumprod,alphas

    def get_alpha_sigma(self,t,use_teacher=False,shape=None):
        if use_teacher:
            alphas_cumprod = self.teacher_alphas_cumprod
        else:
            alphas_cumprod = self.alphas_cumprod
        alpha = tf.gather(tf.sqrt(alphas_cumprod),t)
        sigma = tf.gather(tf.sqrt(1-alphas_cumprod),t)
        if shape is not None:
            alpha = tf.reshape(alpha,shape)
            sigma = tf.reshape(sigma,shape)
        return alpha,sigma

    @tf.function
    def train_step(self, inputs):
        voxel,layer,cond = inputs

        random_t = 2*tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        eps = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)

        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True,shape=self.shape) 
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,shape=self.shape)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True,shape=self.shape)
        
            
        #voxel                        
        z = alpha*voxel + eps * sigma
        score = self.teacher_voxel([z, random_t+1,layer,cond],training=False)
        
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_voxel([z_1, random_t,layer,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2

        w = tf.maximum(1.0,(alpha_s/sigma_s)**2)
        with tf.GradientTape() as tape:
            score = self.model_voxel([z, random_t//2,layer,cond])
            loss_voxel = tf.square(score - target)     
            loss_voxel = tf.reduce_mean(loss_voxel)
            
            
        g = tape.gradient(loss_voxel, self.model_voxel.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_voxel.trainable_variables))
        for weight, ema_weight in zip(self.model_voxel.weights, self.ema_voxel.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)




        #layer
        eps = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True)
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True)
        
            
        z = alpha*layer + eps * sigma
        score = self.teacher_layer([z, random_t+1,cond],training=False)
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_layer([z_1, random_t,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2
        
        w = tf.maximum(1.0,(alpha_s/sigma_s)**2)
        with tf.GradientTape() as tape:
            score = self.model_layer([z, random_t//2,cond])
            loss_layer = tf.square(score - target)
            loss_layer = tf.reduce_mean(loss_layer)
            
            
        g = tape.gradient(loss_layer, self.model_layer.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_layer.trainable_variables))
        for weight, ema_weight in zip(self.model_layer.weights, self.ema_layer.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        
        self.loss_tracker.update_state(loss_voxel+loss_layer)

        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":loss_voxel,
            "loss_layer":loss_layer,
        }

    @tf.function
    def test_step(self, inputs):
        voxel,layer,cond = inputs

        random_t = 2*tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        eps = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)

        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True,shape=self.shape) 
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,shape=self.shape)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True,shape=self.shape)
        
            
        #voxel                        
        z = alpha*voxel + eps * sigma
        score = self.teacher_voxel([z, random_t+1,layer,cond],training=False)
        
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_voxel([z_1, random_t,layer,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2

        w = tf.maximum(1.0,(alpha_s/sigma_s)**2)
        score = self.model_voxel([z, random_t//2,layer,cond])
        loss_voxel = tf.square(score - target)
        loss_voxel = tf.reduce_mean(loss_voxel)
            
        #layer
        eps = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True)
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True)
        
            
        z = alpha*layer + eps * sigma
        score = self.teacher_layer([z, random_t+1,cond],training=False)
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_layer([z_1, random_t,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2
        w = tf.maximum(1.0,(alpha_s/sigma_s)**2)
        
        score = self.model_layer([z, random_t//2,cond])
        loss_layer = tf.square(score - target)
        loss_layer = tf.reduce_mean(loss_layer)
        
        self.loss_tracker.update_state(loss_voxel+loss_layer)

        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":loss_voxel,
            "loss_layer":loss_layer,
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions)

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

        
