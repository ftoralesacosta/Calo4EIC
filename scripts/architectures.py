import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow_addons as tfa

def Unet(
        num_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        pad=((2,1),(0,0),(4,3)),
        use_1D=False,
        #pad=((1,0),(0,0),(1,0)),
):
    #https://github.com/beresandras/clear-diffusion-keras/blob/master/architecture.py
    #act = layers.LeakyReLU(alpha=0.01)
    act = tf.keras.activations.swish
    def ResidualBlock(width, attention):
        def forward(x):
            x , n = x
            input_width = x.shape[2] if use_1D else x.shape[4]
            if input_width == width:
                residual = x
            else:
                if use_1D:
                    residual = layers.Conv1D(width, kernel_size=1)(x)
                else:
                    residual = layers.Conv3D(width, kernel_size=1)(x)

            n = layers.Dense(width)(n)
            # x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv3D(width, kernel_size=kernel, padding="same")(x)
            x = layers.Add()([x, n])
            # x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv3D(width, kernel_size=kernel, padding="same")(x)
            x = layers.Add()([residual, x])

            if attention:
                residual = x
                if use_1D:                    
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1)
                    )(x, x)
                else:
                    x = tfa.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1, 2, 3)
                    )(x, x)

                x = layers.Add()([residual, x])
            return x
        return forward

    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x,n])
                skips.append(x)        
            if use_1D:
                x = layers.AveragePooling1D(pool_size=stride)(x)
            else:
                x = layers.AveragePooling3D(pool_size=stride)(x)
            return x

        return forward

    def UpBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            if use_1D:
                x = layers.UpSampling1D(size=stride)(x)
            else:
                x = layers.UpSampling3D(size=stride)(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width, attention)([x,n])
            return x

        return forward

    inputs = keras.Input((num_dim))
    if use_1D:
        #No padding to 1D model
        x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs)
        n = layers.Reshape((1,time_embedding.shape[-1]))(time_embedding)
    else:
        inputs_padded = layers.ZeroPadding3D(pad)(inputs)
        x = layers.Conv3D(input_embedding_dims, kernel_size=1)(inputs_padded)
        n = layers.Reshape((1,1,1,time_embedding.shape[-1]))(time_embedding)
    
    skips = []
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, n, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x,n])

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, n,  skips])

    if use_1D:
        outputs = layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)
    else:
        outputs = layers.Conv3D(1, kernel_size=1, kernel_initializer="zeros")(x)
        outputs = layers.Cropping3D(pad)(outputs)


    return inputs, outputs



def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
        activation='leakyrelu'
):

    
    act = layers.LeakyReLU(alpha=0.01)
    #act = tf.keras.activations.swish

    def resnet_dense(input_layer,hidden_size):
        layer,time = input_layer
        residual = layers.Dense(hidden_size)(layer)
        embed =  layers.Dense(hidden_size)(time)
        x = act(layer)
        x = layers.Dense(hidden_size)(x)
        x = act(layers.Add()([x, embed]))
        x = layers.Dense(hidden_size)(x)
        x = layers.Add()([x, residual])
        return x

    embed = act(layers.Dense(mlp_dim)(time_embedding))
    
    layer = layers.Dense(mlp_dim)(inputs)
    for _ in range(num_layer-1):
        layer =  resnet_dense([layer,embed],mlp_dim)

    outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)



    
    # def resnet_dense(input_layer,hidden_size,nlayers=2):
    #     layer = input_layer
    #     residual = layers.Dense(hidden_size)(layer)
    #     for _ in range(nlayers):
    #         layer = act(layers.Dense(hidden_size,activation=None)(layer))
    #         layer = layers.Dropout(0.1)(layer)
    #     return residual + layer
    
    # embed = layers.Dense(mlp_dim)(time_embedding)
    # residual = act(layers.Dense(2*mlp_dim)(tf.concat([inputs,embed],-1)))    
    # residual = layers.Dense(mlp_dim)(residual)
    # layer = residual
    # for _ in range(num_layer-1):
    #     layer =  resnet_dense(layer,mlp_dim)

    # layer = act(layers.Dense(mlp_dim)(residual+layer))
    # outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)
    
    return outputs








def time_conv(input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True,data_shape=(1,1)):
    ## Incorporate information from conditional inputs
    time_layer = layers.Dense(hidden_size,activation="swish",use_bias=False)(embed)
    
    if len(data_shape) == 2:
        layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                              strides=1,use_bias=False,activation='swish')(input_layer)
        time_layer = tf.reshape(time_layer,(-1,1,hidden_size))
        layer=layer+time_layer
        layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,
                              padding=padding,activation='swish',
                              strides=1,use_bias=True)(layer) 
        
    else:
        layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                              strides=1,use_bias=False,activation='swish')(input_layer)
        time_layer = tf.reshape(time_layer,(-1,1,1,1,hidden_size))
        layer=layer+time_layer
        layer = layers.Conv3D(hidden_size,kernel_size=1,
                              padding=padding,activation='swish',
                              strides=1,use_bias=True)(layer)
    return layer    


def ConvModel(
        time_embed,
        data_shape,
        conv_sizes,
        stride_size,
        kernel_size,
        nlayers,
        activation='swish',

):     
        inputs = Input((data_shape))
        def ConvBlocks(layer,conv_sizes,stride_size,kernel_size,nlayers):
            skip_layers = []
            
            layer_encoded = time_conv(layer,time_embed,conv_sizes[0],
                                      kernel_size=kernel_size,
                                      stride=1,padding='same',data_shape=data_shape)
            skip_layers.append(layer_encoded)
            #print(layer_encoded)
            for ilayer in range(1,nlayers):
                layer_encoded = time_conv(skip_layers[-1],time_embed,conv_sizes[ilayer],
                                          kernel_size=kernel_size,padding='same',
                                          #stride=stride_size,
                                          stride=1,data_shape=data_shape,
                )
                
                if len(data_shape) == 2:
                    layer_encoded = layers.AveragePooling1D(stride_size)(layer_encoded)
                else:
                    layer_encoded = layers.AveragePooling3D(stride_size)(layer_encoded)
                skip_layers.append(layer_encoded)

            return skip_layers[::-1]

        def ConvTransBlocks(skip_layers,conv_sizes,stride_size,kernel_size):
            layer_decoded = time_conv(skip_layers[0],
                                      time_embed,conv_sizes[len(skip_layers)-1],
                                      stride = 1,data_shape=data_shape,
                                      kernel_size=kernel_size,padding='same')
            for ilayer in range(len(skip_layers)-1):
                layer_decoded = time_conv(layer_decoded,
                                          time_embed,conv_sizes[len(skip_layers)-2-ilayer],
                                          stride = 1,data_shape=data_shape,
                                          kernel_size=kernel_size,padding='same')
                if len(data_shape) == 2:
                    layer_decoded = layers.UpSampling1D(stride_size)(layer_decoded)
                    layer_decoded =layers.Conv1D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=kernel_size,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=activation)(layer_decoded)
                else:
                    layer_decoded = layers.UpSampling3D(stride_size)(layer_decoded)
                    layer_decoded =layers.Conv3D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=kernel_size,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=activation)(layer_decoded)
                    
                
                layer_decoded = (layer_decoded+ skip_layers[ilayer+1])/np.sqrt(2)
                if len(data_shape) == 2:
                    layer_decoded =layers.Conv1D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=1,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=activation)(layer_decoded)
                else:
                    layer_decoded =layers.Conv3D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=1,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=activation)(layer_decoded)
            layer_decoded = time_conv(layer_decoded,
                                      time_embed,conv_sizes[0],
                                      stride = 1,data_shape=data_shape,
                                      kernel_size=kernel_size,padding='same')
            return layer_decoded

        

        cnn_encoder = ConvBlocks(inputs,conv_sizes,stride_size=stride_size,kernel_size = kernel_size,nlayers = nlayers)
        cnn_decoder = ConvTransBlocks(cnn_encoder,conv_sizes,stride_size=stride_size,kernel_size = kernel_size)
        if len(data_shape) == 2:
            outputs = layers.Conv1D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)
        else:
            outputs = layers.Conv3D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)
        
        return inputs, outputs
        
