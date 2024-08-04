import tensorflow as tf
import tensorflow_addons as tfa
from libml import utils
from absl import flags

FLAGS = flags.FLAGS

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

class residual_basic(tf.keras.Model):
    def __init__(self, filters, stride, name="", sc=False, is_last=False):
        super(residual_basic, self).__init__(name=name)
        self.sc = sc
        self.is_last = is_last
        self.filters = filters

        self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='SAME',
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.fc2a = tf.keras.layers.Dense(units=filters, use_bias=False)
        self.bn2a = tf.keras.layers.BatchNormalization(momentum=0.99)#tfa.layers.GroupNormalization(groups=2, axis=3)#tf.keras.layers.BatchNormalization(momentum=0.999)
        self.drops = tf.keras.layers.Dropout(rate=0.5)

        self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.bn2b = tf.keras.layers.BatchNormalization(momentum=0.99)#tfa.layers.GroupNormalization(groups=2, axis=3)#tf.keras.layers.BatchNormalization(momentum=0.999)
        self.fc2b = tf.keras.layers.Dense(units=filters, use_bias=False)
        if sc:
            self.conv2c = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='SAME',
                                                 kernel_initializer=tf.keras.initializers.he_normal())
            self.fc2c = tf.keras.layers.Dense(units=filters, use_bias=False)
        self.bn2c = tf.keras.layers.BatchNormalization(momentum=0.99)#tfa.layers.GroupNormalization(groups=2, axis=3)#tf.keras.layers.BatchNormalization(momentum=0.999)

    def call(self, x0, training=False):
        x = self.conv2a(x0)# + tf.reshape(self.fc2a(dm), [-1, 1, 1, self.filters])
        x = tf.nn.relu(x) #self.bn2a(x, training=training)
        x = self.bn2a(x, training=training)
        x = self.drops(x, training=training)

        x = self.conv2b(x)
        x = tf.nn.relu(x) #self.bn2b(x, training=training)
        x = self.bn2b(x, training=training)

        if self.sc:
            x0 = self.conv2c(x0)
        x += x0
        if not self.is_last:
            x = tf.nn.relu(x) # self.bn2c(x, training=training)
            x = self.bn2c(x, training=training)
        else:
            x = tf.nn.tanh(x) #self.bn2c(x, training=training)
            x = self.bn2c(x, training=training)

        return x

class init_conv(tf.keras.Model):
    def __init__(self):
        super(init_conv, self).__init__(name='initial_conv')
        self.zp1 = tf.keras.layers.ZeroPadding2D(((2, 3), (2, 3)))
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=(2, 2),
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.fc2a = tf.keras.layers.Dense(units=16, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99)#tfa.layers.GroupNormalization(groups=2, axis=3)#tf.keras.layers.BatchNormalization(momentum=0.99)
        # self.zp2 = tf.keras.layers.ZeroPadding2D((1, 1))
        # self.mp = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='VALID')

    def call(self, x0, training=False):
        x = self.conv1(x0)# + tf.reshape(self.fc2a(dm), [-1, 1, 1, 16])
        # x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.bn(x, training=training)

        return x

class LecunLCN(tf.keras.Model):
    def __init__(self):
        super(LecunLCN, self).__init__(name='LecunLCN')
        self.kernel = tf.cast(gaussian_kernel(9, 0.0, 1.0), 'float32')#tf.ones([25, 25, 1, 1], dtype='float16')#tf.Variable(initial_value=tf.ones([36, 36, 1, 1], dtype='float16'),
                                  #trainable=True)
        self.kernel = self.kernel[:, :, tf.newaxis, tf.newaxis]

        # self.kernel /= tf.reduce_sum(self.kernel)


    def call(self, x0, mask):
        red, blue, green = tf.split(x0, [1, 1, 1], axis=3)
        # kernel = self.kernel / tf.reduce_sum(self.kernel)

        ## Channel 0
        mean = tf.nn.conv2d(red, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        red = red - mean
        var = tf.nn.conv2d(red ** 2, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        std = (tf.sqrt(var) + 1e-4)
        red /= std
        red = red * mask

        ## Channel 1
        mean = tf.nn.conv2d(blue, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        blue = blue - mean
        var = tf.nn.conv2d(blue ** 2, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        std = (tf.sqrt(var) + 1e-4)
        blue /= std
        blue = blue * mask

        ## Channel 2
        mean = tf.nn.conv2d(green, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        green = green - mean
        var = tf.nn.conv2d(green ** 2, self.kernel, padding='SAME', strides=[1, 1, 1, 1])
        std = (tf.sqrt(var) + 1e-4)
        green /= std
        green = green * mask

        x = tf.concat([red, blue, green], axis=3)

        return x

def dist_loss(y_pred, y_true, batch_size):
    margin = 1.0
    total_loss = tf.reduce_sum(tf.math.maximum(tf.where(tf.greater(y_true[:batch_size // 2], y_true[batch_size // 2:]),
                                                        x=-(y_pred[:batch_size // 2] - y_pred[batch_size // 2:]) + margin,
                                                        y=(y_pred[:batch_size // 2] - y_pred[batch_size // 2:]) + margin), 0)) / batch_size * 2

    return total_loss


def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.compat.v1.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = 'float32'))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """

    return tf.nn.swish(x)


def block(inputs, activation_fn=swish, drop_rate=0., name='',
        filters_in=32, filters_out=16, kernel_size=3, strides=1,
        expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3# if backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(filters, 1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'expand_conv')(inputs)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = tf.keras.layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = tf.keras.layers.DepthwiseConv2D(kernel_size,
                            strides=strides,
                            padding=conv_pad,
                            use_bias=False,
                            depthwise_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'dwconv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = tf.keras.layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = tf.keras.layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = tf.keras.layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = tf.keras.layers.Conv2D(filters_se, 1,
                        padding='same',
                        activation=activation_fn,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'se_reduce')(se)
        se = tf.keras.layers.Conv2D(filters, 1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'se_expand')(se)

        x = tf.keras.layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = tf.keras.layers.Conv2D(filters_out, 1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=name + 'project_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate,
                            noise_shape=(None, 1, 1, 1),
                            name=name + 'drop')(x)
        x = tf.keras.layers.add([x, inputs], name=name + 'add')

    return x


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))