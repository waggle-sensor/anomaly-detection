import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.regularizers import l2

class Attention_CAE(Model):
        
    def __init__(self, latent_dim=64, input_shape=(960,1280,3), 
                 attention_weight=0.001):
        super(Attention_CAE, self).__init__()
        
        # The number of encoding features
        self.latent_dim = latent_dim
        
        # weight for attention regularization:
        self.attention_weight = attention_weight
        
        # encoder leaky ReLU
        lrelu = layers.LeakyReLU(alpha=0.1)
        
        # encoder regularizer
        reg = l2(5e-4)
        
        # Encoder Layers:
        self.conv_encoder = tf.keras.Sequential([
            # apply 7x7 filter with maxpooling:
            layers.Conv2D(filters=16, kernel_size= (11, 11), 
                            strides=(2, 2), input_shape =input_shape, 
                            padding = 'same', activation=layers.LeakyReLU(alpha=0.1), 
                            kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),
            
            # apply 3x3 filter with maxpooling:
            layers.Conv2D(filters=32, kernel_size= (3, 3), 
                            padding = 'same', activation=lrelu, 
                            kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),
            
            # apply 3x3, 1x1 filter with maxpooling:
            layers.Conv2D(filters=48, kernel_size= (1, 1), 
                            padding = 'same', activation=lrelu, 
                            kernel_regularizer=reg),
            layers.Conv2D(filters=48, kernel_size= (3, 3), 
                            padding = 'same', activation=lrelu, 
                            kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),
            
            # apply 3x3, 1x1 filter (do maxpooling in the latent encoder):
            layers.Conv2D(filters=64, kernel_size= (1, 1), 
                            padding = 'same', activation=lrelu, 
                            kernel_regularizer=reg),
            layers.Conv2D(filters=64, kernel_size= (3, 3), 
                            padding = 'same', activation=lrelu, 
                            kernel_regularizer=reg, name='last_conv_layer'),
        ],name='conv_encoder')
        
        self.last_conv_output_shape = tuple(x for x in self.conv_encoder.layers[-1].output_shape if x != None)
        self.latent_encoder = tf.keras.Sequential([
            # pool output from last conv layer:
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same',
                                input_shape=self.last_conv_output_shape),
            
            # add dense latent layer:
            layers.Flatten(),
            layers.Dense(latent_dim)
        ], name='latent_encoder')
        
        # Decoder Layers:
        last_maxpool_output_shape = tuple( x for x in self.latent_encoder.layers[-2].input_shape if x != None )
        last_maxpool_output_size = np.prod(last_maxpool_output_shape)
        self.decoder = tf.keras.Sequential([
            # reshape back to last conv layer output:
            layers.Dense(last_maxpool_output_size, input_shape=(latent_dim,)),
            layers.Reshape(last_maxpool_output_shape),
            
            # apply upsampling, then transposed 1x1 and 2x2 filter (no regularizer):
            layers.UpSampling2D(size=(2,2)),
            layers.Conv2DTranspose(filters=64, kernel_size=(3,3),
                                      padding='same', activation=lrelu),
            layers.Conv2DTranspose(filters=64, kernel_size=(1,1),
                                      padding='same', activation=lrelu),
            
            # apply upsampling, then transposed 1x1 and 2x2 filter (no regularizer):
            layers.UpSampling2D(size=(2,2)),
            layers.Conv2DTranspose(filters=48, kernel_size=(3,3),
                                      padding='same', activation=lrelu),
            layers.Conv2DTranspose(filters=48, kernel_size=(1,1),
                                      padding='same', activation=lrelu),
            
            # apply upsampling, then transposed 3x3 filter (no regularizer):
            layers.UpSampling2D(size=(2,2)),
            layers.Conv2DTranspose(filters=32, kernel_size=(3,3),
                                      padding='same', activation=lrelu),
            
            # apply upsampling, then transposed 11x11 filter (no regularizer):
            layers.UpSampling2D(size=(2,2)),
            layers.Conv2DTranspose(filters=16, kernel_size=(11,11), strides=(2, 2),
                                      padding='same', activation=lrelu),
            layers.Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='sigmoid')
        ], name='decoder')
    
    
    def compute_loss(self, x):
        x_pred, attention_map = self.call_with_attention(x)
        loss = tf.reduce_mean(tf.square(x_pred - x))
        ae_loss = tf.reduce_mean(0.25 - tf.tanh(attention_map))
        
        return loss + self.attention_weight*ae_loss
        
    def call_with_attention(self, x):
        conv_output = self.conv_encoder(x)
        with tf.GradientTape() as tape:
            tape.watch(conv_output)
            z = self.latent_encoder(conv_output)
            grads = tape.gradient(z, conv_output)
            
        # compute the attention map by multiplying the convolutional 
        # layer output by the pooled gradients and taking the positive 
        # component of the summation over all channels 
        # (this is the Grad-CAM algorithm)
        pooled_grads = tf.reduce_mean(grads, axis=[1,2], keepdims=True)
        attention_channels = tf.multiply(pooled_grads, conv_output)
        attention_map = tf.nn.relu(tf.reduce_mean(attention_channels, axis=-1))
        
        decoded = self.decoder(z)
        return decoded, attention_map
    
    def call(self, x):
        z = self.latent_encoder(self.conv_encoder(x))
        x_pred = self.decoder(z)
        return tf.reduce_sum(tf.square(x_pred - x), axis=-1)
    
    @tf.function
    def train_step(self, x, optimizer):
        """Executes a single training step"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))
            return loss