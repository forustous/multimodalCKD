import tensorflow as tf
import itertools
import math

from libml.layers import residual_basic, init_conv, LecunLCN, swish, block, CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER, correct_pad
from absl import flags

FLAGS = flags.FLAGS
IMAGE_SIZE = 512


DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]


class MultiMoel():
    def WRN_40_2(self, filters):
        inputs          = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
        age             = tf.keras.layers.Input(shape=1, name='age')
        sex             = tf.keras.layers.Input(shape=1, name='sex')
        blood           = tf.keras.layers.Input(shape=1, name='blood')
        bilirubin       = tf.keras.layers.Input(shape=1, name='bilirubin')
        urobilinogen    = tf.keras.layers.Input(shape=1, name='urobilinogen')
        ketone          = tf.keras.layers.Input(shape=1, name='ketone')
        protein         = tf.keras.layers.Input(shape=1, name='protein')
        nitrite         = tf.keras.layers.Input(shape=1, name='nitrite')
        glucose         = tf.keras.layers.Input(shape=1, name='glucose')
        leucocyte       = tf.keras.layers.Input(shape=1, name='leucocyte')
        ph              = tf.keras.layers.Input(shape=1, name='ph')
        sg              = tf.keras.layers.Input(shape=1, name='sg')

        urine = tf.concat([age, sex, protein, ketone, glucose, ph, blood, bilirubin, urobilinogen, nitrite, leucocyte, sg], axis=1)

        repeat = [4, 4, 4]

        conv0 = init_conv()
        LCN   = LecunLCN()

        x     = tf.keras.Sequential([conv0])(inputs)
        
        feat = x

        for scale in range(len(repeat)):
            x = residual_basic(filters << scale, name="{}".format(scale), stride=(2, 2), sc=True)(x)
            for i in range(repeat[scale] - 1):
                if scale == 2 and i == 3:
                    x = residual_basic(filters << scale, name="{}_{}".format(scale, i), stride=(1, 1), is_last=True)(x)
                else:
                    x = residual_basic(filters << scale, name="{}_{}".format(scale, i), stride=(1, 1))(x)
                
        feats0  = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Add urine analysis
        urine = tf.cast(urine, 'float32')
        feats0 = tf.concat([feats0, urine], axis=1)

        # # Predictions for eGFR
        outputs60   = tf.keras.layers.Dense(units=1, name='predictions', activation='sigmoid', dtype=tf.float32)(feats0)

        model = tf.keras.Model(inputs=[inputs, age, sex, blood, bilirubin, urobilinogen, ketone, protein, nitrite, glucose, leucocyte, ph, sg],
                               outputs={'gt60' : outputs60,
                                        # 'A_K': feat})
                                        'A_K': inputs})

        return model

    def EfficientNet(self, width_coefficient,
                    depth_coefficient,
                    default_size,
                    dropout_rate=0.2,
                    drop_connect_rate=0.2,
                    depth_divisor=8,
                    activation_fn=swish,
                    blocks_args=DEFAULT_BLOCKS_ARGS,
                    model_name='efficientnet',
                    include_top=True,
                    pooling=None,
                    classes=1,
                    **kwargs):


        inputs          = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
        age             = tf.keras.layers.Input(shape=1, name='age')
        sex             = tf.keras.layers.Input(shape=1, name='sex')
        blood           = tf.keras.layers.Input(shape=1, name='blood')
        bilirubin       = tf.keras.layers.Input(shape=1, name='bilirubin')
        urobilinogen    = tf.keras.layers.Input(shape=1, name='urobilinogen')
        ketone          = tf.keras.layers.Input(shape=1, name='ketone')
        protein         = tf.keras.layers.Input(shape=1, name='protein')
        nitrite         = tf.keras.layers.Input(shape=1, name='nitrite')
        glucose         = tf.keras.layers.Input(shape=1, name='glucose')
        leucocyte       = tf.keras.layers.Input(shape=1, name='leucocyte')
        ph              = tf.keras.layers.Input(shape=1, name='ph')
        sg              = tf.keras.layers.Input(shape=1, name='sg')

        urine = tf.concat([age, sex, protein, ketone, glucose, ph, blood, bilirubin, urobilinogen, nitrite, leucocyte, sg], axis=1)

        bn_axis = 3

        def round_filters(filters, divisor=depth_divisor):
            """Round number of filters based on depth multiplier."""
            filters *= width_coefficient
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        # Build stem
        x = tf.keras.layers.Resizing(default_size, default_size)(inputs)
        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                                name='stem_conv_pad')(x)

        x = tf.keras.layers.Conv2D(round_filters(32), 3,
                        strides=2,
                        padding='valid',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='stem_conv')(x)
        
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
        x = tf.keras.layers.Activation(activation_fn, name='stem_activation')(x)
        # Build blocks
        from copy import deepcopy
        blocks_args = deepcopy(blocks_args)

        # feat = x    # feature for CAM
        
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = block(x, activation_fn, drop_connect_rate * b / blocks,
                        name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
                b += 1

                if i==5 and j==0:
                  feat = x


        # Build top
        x = tf.keras.layers.Conv2D(round_filters(1280), 1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='top_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
        x = tf.keras.layers.Activation(activation_fn, name='top_activation')(x)

        if include_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
            
            urine = tf.cast(urine, 'float32')
            x = tf.concat([x, urine], axis=1)
            
            x = tf.keras.layers.Dense(classes,
                            activation='sigmoid',
                            kernel_initializer=DENSE_KERNEL_INITIALIZER,
                            name='probs')(x)
        else:
            if pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

        model = tf.keras.Model(inputs=[inputs, age, sex, blood, bilirubin, urobilinogen, ketone, protein, nitrite, glucose, leucocyte, ph, sg],#, ], #
                               outputs={'gt60' : x,
                                        'A_K': feat})
        return model


    def multi_model(self, filters, n_classes):
        if FLAGS.arch == 'WRN-40-2':
            return self.WRN_40_2(filters)
        elif FLAGS.arch == 'Efficient':
            # https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
            # B0 : 1.0, 1.0, 224, 0.2
            # B1 : 1.0, 1.1, 240, 0.2
            # B2 : 1.1, 1.2, 260, 0.3
            # B3 : 1.2, 1.4, 300, 0.3
            # B4 : 1.4, 1.8, 380, 0.4
            # B5 : 1.6, 2.2, 456, 0.4
            # B6 : 1.8, 2.6, 528, 0.5
            # B7 : 2.0, 3.1, 600, 0.5
            return self.EfficientNet(1.0, 1.0, 224, 0.2,#1.1, 1.2, 260, 0.3, model_name='efficientnet-b2',
                        model_name='efficientnet-b1',
                        classes=1)
        else:
            raise("Architecture is not supported")

flags.DEFINE_string('arch', 'Efficient', 'CNN Architectures')
