import tensorflow as tf
import keras
from keras_self_attention import SeqSelfAttention

OUTPUT_CHANNELS = 3


class _Attention(tf.keras.layers.Layer):
    
    def __init__(self, data_format='channels_last', **kwargs):
        super(_Attention, self).__init__(**kwargs)
        self.data_format = data_format
        
    def build(self, input_shapes):
        self.gamma = self.add_weight(self.name + '_gamma',
                                     shape=(),
                                     initializer=tf.initializers.Zeros)
    
    def call(self, inputs):
        if len(inputs) != 4:
            raise Exception('an attention layer should have 4 inputs')

        query_tensor = inputs[0]
        key_tensor =  inputs[1]
        value_tensor = inputs[2]
        origin_input = inputs[3]
        
        input_shape = tf.shape(query_tensor)
        
        if self.data_format == 'channels_first':
            height_axis = 2
            width_axis = 3
        else:
            height_axis = 1
            width_axis = 2
        
        batchsize = input_shape[0]
        height = input_shape[height_axis]
        width = input_shape[width_axis]
        
        if self.data_format == 'channels_first':
            proj_query = tf.transpose(
                tf.reshape(query_tensor, (batchsize, -1, height*width)),(0, 2, 1))
            proj_key = tf.reshape(key_tensor, (batchsize, -1, height*width))
            proj_value = tf.reshape(value_tensor, (batchsize, -1, height*width))
        else:
            proj_query = tf.reshape(query_tensor, (batchsize, height*width, -1))
            proj_key = tf.transpose(
                tf.reshape(key_tensor, (batchsize, height*width, -1)), (0, 2, 1))
            proj_value = tf.transpose(
                tf.reshape(value_tensor, (batchsize, height*width, -1)), (0, 2, 1))

        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))
        
        if self.data_format == 'channels_first':
            out = tf.reshape(out, (batchsize, -1, height, width))
        else:
            out = tf.reshape(
                tf.transpose(out, (0, 2, 1)), (batchsize, height, width, -1))
        
        return tf.add(tf.multiply(out, self.gamma), origin_input)#, attention
    
class SelfAttnModel(tf.keras.Model):
    def __init__(self, input_dims, data_format='channels_last', **kwargs):
        super(SelfAttnModel, self).__init__(**kwargs)
        self.attn = _Attention(data_format=data_format)
        self.query_conv = tf.keras.layers.Conv2D(filters=input_dims//8,
                                        kernel_size=1,
                                        data_format=data_format)
        self.key_conv = tf.keras.layers.Conv2D(filters=input_dims//8,
                                      kernel_size=1,
                                      data_format=data_format)
        self.value_conv = tf.keras.layers.Conv2D(filters=input_dims,
                                        kernel_size=1,
                                        data_format=data_format)
    
    def call(self, inputs, training=False):
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        return self.attn([q, k, v, inputs])

def downsample(filters, size, shape, apply_batchnorm=True, attn_layer=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', batch_input_shape=shape, 
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    if attn_layer:
        result.add(SelfAttnModel(input_dims=filters))
#     result.add(SeqSelfAttention(attention_activation='sigmoid'))

    return result

def upsample(filters, size, shape, apply_dropout=False, attn_layer=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2, batch_input_shape=shape,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    
    if attn_layer:
        result.add(SelfAttnModel(input_dims=filters))

    return result

def generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, (None, 256, 256, 3), apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4, (None, 128, 128, 64)), # (bs, 64, 64, 128)
        downsample(256, 4, (None, 64, 64, 128)), # (bs, 32, 32, 256)
        downsample(512, 4, (None, 32, 32, 256)), # (bs, 16, 16, 512)
        downsample(512, 4, (None, 16, 16, 512)), # (bs, 8, 8, 512)
        downsample(512, 4, (None, 8, 8, 512)), # (bs, 4, 4, 512)
        downsample(512, 4, (None, 4, 4, 512)), # (bs, 2, 2, 512)
        downsample(512, 4, (None, 2, 2, 512)), # (bs, 1, 1, 512)
    ]
    skip_down_stack = [
        downsample(64, 4, (None, 256, 256, 3), apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4, (None, 128, 128, 64)), # (bs, 64, 64, 128)
        downsample(256, 4, (None, 64, 64, 128)), # (bs, 32, 32, 256)
        downsample(512, 4, (None, 32, 32, 256), attn_layer=True), # (bs, 16, 16, 512)
        downsample(512, 4, (None, 16, 16, 512)), # (bs, 8, 8, 512)
        downsample(512, 4, (None, 8, 8, 512)), # (bs, 4, 4, 512)
        downsample(512, 4, (None, 4, 4, 512)), # (bs, 2, 2, 512)
        downsample(512, 4, (None, 2, 2, 512)), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, (None, 1, 1, 512), apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, (None, 2, 2, 1024), apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, (None, 4, 4, 1024), apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4, (None, 8, 8, 1024)), # (bs, 16, 16, 1024)
        upsample(256, 4, (None, 16, 16, 1024)), # (bs, 32, 32, 512)
        upsample(128, 4, (None, 32, 32, 512)), # (bs, 64, 64, 256)
        upsample(64, 4, (None, 64, 64, 256), attn_layer=True), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
#     for sdown in skip_down_stack:
#         skips.append(down(x))

    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
