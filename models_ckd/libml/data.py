import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import glob

from libml import utils
from absl import flags

flags.DEFINE_integer('random_seed', 0, 'Seed.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 16384, 'Size of dataset shuffling.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS

IMAGE_SIZE = 512#587

def record_parse(serialized_example):
    
    if FLAGS.hospital_source_type_for_prediction=='SCHPC':
      features = tf.io.parse_single_example(
          serialized_example,
          features={'image'       : tf.io.FixedLenFeature([], tf.string),
                    'img_id'      : tf.io.FixedLenFeature([], tf.string),
                    'mask'        : tf.io.FixedLenFeature([], tf.string),
                    'CKD'         : tf.io.FixedLenFeature([], tf.int64),
                    'eGFR'        : tf.io.FixedLenFeature([], tf.int64),
                    'eGFR45'      : tf.io.FixedLenFeature([], tf.int64),
                    'age'         : tf.io.FixedLenFeature([], tf.int64),
                    'sex'         : tf.io.FixedLenFeature([], tf.int64),
                    'blood'       : tf.io.FixedLenFeature([], tf.int64),
                    'bilirubin'   : tf.io.FixedLenFeature([], tf.int64),
                    'urobilinogen': tf.io.FixedLenFeature([], tf.int64),
                    'ketone'      : tf.io.FixedLenFeature([], tf.int64),
                    'protein'     : tf.io.FixedLenFeature([], tf.int64),
                    'nitrite'     : tf.io.FixedLenFeature([], tf.int64),
                    'glucose'     : tf.io.FixedLenFeature([], tf.int64),
                    'ph'          : tf.io.FixedLenFeature([], tf.float32),
                    'sg'          : tf.io.FixedLenFeature([], tf.float32),
                    'leucocyte'   : tf.io.FixedLenFeature([], tf.int64)
                    })
    elif FLAGS.hospital_source_type_for_prediction=='CHA':
      features = tf.io.parse_single_example(
          serialized_example,
          features={'image'       : tf.io.FixedLenFeature([], tf.string),
                    'img_id'      : tf.io.FixedLenFeature([], tf.string),
                    'mask'        : tf.io.FixedLenFeature([], tf.string),
                    'CKD'         : tf.io.FixedLenFeature([], tf.int64),
                    'eGFR'        : tf.io.FixedLenFeature([], tf.int64),
                    'eGFR45'      : tf.io.FixedLenFeature([], tf.int64),
                    'age'         : tf.io.FixedLenFeature([], tf.int64),
                    'sex'         : tf.io.FixedLenFeature([], tf.int64),
                    'blood'       : tf.io.FixedLenFeature([], tf.int64),
                    'bilirubin'   : tf.io.FixedLenFeature([], tf.int64),
                    'urobilinogen': tf.io.FixedLenFeature([], tf.int64),
                    'ketone'      : tf.io.FixedLenFeature([], tf.int64),
                    'protein'     : tf.io.FixedLenFeature([], tf.int64),
                    'nitrite'     : tf.io.FixedLenFeature([], tf.int64),
                    'glucose'     : tf.io.FixedLenFeature([], tf.int64),
                    'ph'          : tf.io.FixedLenFeature([], tf.float32),
                    'sg'          : tf.io.FixedLenFeature([], tf.float32),
                    'leucocyte'   : tf.io.FixedLenFeature([], tf.int64)
                    })
    else:
      print('check hospital_source_type_for_prediction')


    image = tf.io.decode_raw(features['image'], out_type=tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = image[:, :, ::-1]
    image = tf.cast(image, tf.float32)
    # if FLAGS.hospital_source_type_for_prediction=='CHA':
    #   img_id=features['img_id']
    img_id=features['img_id']  
    ori_image = image
    #image = tf.cast(image, tf.float32) / 255.0
    image = image - tf.reduce_mean(image, axis=(0,1))
    image = image / tf.sqrt(tf.reduce_mean(image ** 2, axis=(0,1)))

    # age         = [(tf.cast(features['age'], tf.float32) - 35) / (80-35)]
    age         = [(tf.cast(features['age'], tf.float32) - 45.4695) / 11.7626]
    sex         = [features['sex']]
    sex         = [tf.cast(features['sex'], 'float32')]

    # eGFR        = features['CKD']
    # eGFR45      = features['eGFR45']
    # label       = [features['eGFR']]
    label       = [tf.cast(features['eGFR'], 'float32')]
    label45     = [features['eGFR45']]

    # Urine
    # blood        = [(tf.cast(features['blood'], 'float32') - 0) / (5-0)]
    blood        = [(tf.cast(features['blood'], 'float32') - 0.4222) / 0.8937]
    # bilirubin    = [(tf.cast(features['bilirubin'], 'float32') - 0) / (4-0)]
    bilirubin    = [(tf.cast(features['bilirubin'], 'float32') - 0.03446) / 0.2683]
    # urobilinogen = [(tf.cast(features['urobilinogen'], 'float32') - 0) / (4-0)]
    urobilinogen = [(tf.cast(features['urobilinogen'], 'float32') - 0.07676) / 0.4077]
    # ketone       = [(tf.cast(features['ketone'], 'float32') - 0) / (5-0)]
    ketone       = [(tf.cast(features['ketone'], 'float32') - 0.1999) / 0.6776]
    # protein      = [(tf.cast(features['protein'], 'float32') - 0) / (5-0)]
    protein      = [(tf.cast(features['protein'], 'float32') - 0.2496) / 0.5990]
    # nitrite      = [(tf.cast(features['nitrite'], 'float32') - 0) / (1-0)]
    nitrite      = [(tf.cast(features['nitrite'], 'float32') - 0.00912) / 0.09506]
    # glucose      = [(tf.cast(features['glucose'], 'float32') - 0) / (5-0)]
    glucose      = [(tf.cast(features['glucose'], 'float32') - 0.1507) / 0.7654]
    # leucocyte    = [(tf.cast(features['leucocyte'], 'float32') - 0) / (5-0)]
    leucocyte    = [(tf.cast(features['leucocyte'], 'float32') - 0.31601) / 0.8468]
    # ph           = [(tf.cast(features['ph'], 'float32') - 5) / (9-5)]
    ph           = [(tf.cast(features['ph'], 'float32') - 6.1786) / 0.7942]
    # sg           = [(tf.cast(features['sg'], 'float32') - 1.005) / (1.03-1.005)]
    sg           = [(tf.cast(features['sg'], 'float32') - 1.0187) / 0.0089]

    weight     = (tf.cast(features['eGFR'], 'float32') * 29.3869 + 0.5086)
    weight45   = (tf.cast(features['eGFR45'], 'float32') * 84.6777 + 0.5029)

    zeros = tf.constant(0.0, dtype=tf.float32)
    ones  = tf.constant(0.0, dtype=tf.float32)

    if FLAGS.hospital_source_type_for_prediction=='SCHPC':
      return dict(image=image, age=age, blood=blood, bilirubin=bilirubin, urobilinogen=urobilinogen,
                  ketone=ketone, protein=protein, nitrite=nitrite, glucose=glucose, leucocyte=leucocyte, ph=ph, sg=sg,
                  sex=sex, label=label, label45=label45, weight=weight, weight45=weight45, ori_image=ori_image,
                  img_id=img_id
                  )
    elif FLAGS.hospital_source_type_for_prediction=='CHA':
      return dict(image=image, age=age, blood=blood, bilirubin=bilirubin, urobilinogen=urobilinogen,
                  ketone=ketone, protein=protein, nitrite=nitrite, glucose=glucose, leucocyte=leucocyte, ph=ph, sg=sg,
                  sex=sex, label=label, label45=label45, weight=weight, weight45=weight45, ori_image=ori_image,
                  img_id=img_id
                  )


def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
    return dataset.map(parse_fn, num_parallel_calls=para)

def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    return tf.data.TFRecordDataset(filenames)

def augment_mirror(x):
    return tf.image.random_flip_left_right(x)#tf.keras.layers.RandomRotation(factor=(0.1, 0.1))(tf.image.random_flip_left_right(x))

def adjust_contrast(x):
    return tf.image.adjust_contrast(x, 2.0)

def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.image.random_crop(y, tf.shape(x))

def augment_hue(x):
    return tf.image.random_hue(x, 0.1)

class DataSet:
    def __init__(self, name, train, valid, test,
                 height=IMAGE_SIZE, width=IMAGE_SIZE, colors=3, nclass=2, mean=0, std=1):
        self.name   = name
        self.train  = train
        self.valid  = valid
        self.test   = test
        self.height = height
        self.width  = width
        self.colors = colors
        self.nclass = nclass
        self.mean   = mean
        self.std    = std

    @classmethod
    def creator(cls, name, augment, augment_valid, parse_fn=default_parse, colors=3, nclass=2, height=IMAGE_SIZE, width=IMAGE_SIZE):
        fn = lambda x: x.repeat()
        def create():
            # DATA_DIR = './data/devset/tfrecords/'
            # DATA_DIR = '/home/latte/project/retina_ckd/code/eGFR_age_sex_with_proteinuria_EfficientNet_SmoothGrad/data/devset/tfrecords/'
            # DATA_DIR = '/home/latte/project/retina_ckd/code/eGFR_age_sex_with_proteinuria_EfficientNet_SmoothGrad/data_w_id/devset/tfrecords/'
            # DATA_DIR = '/home/latte/project/retina_ckd/code/eGFR_age_sex_with_proteinuria_EfficientNet_SmoothGrad/data_w_id_bright_haze_minmax/devset/tfrecords/'
            DATA_DIR = '/home/young/Github_upload/models_ckd/data/devset/tfrecords/'
            # EXT_DIR  = './data/SCHPC/tfrecords/'
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_with_proteinuria_EfficientNet_SmoothGrad/data/devset/tfrecords/'
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_with_proteinuria_val_with_full_schpc/data/SCHPC/tfrecords/'
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_10ua_train_with_cha_no_transient_protein_process_val_with_full_schpc/data/SCHPC/tfrecords/'
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_10ua_train_with_cha_no_transient_protein_process_val_with_full_new_schpc_code/data/SCHPC/tfrecords/'
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_10ua_train_with_cha_no_transient_protein_process_val_with_full_new_schpc_code/data/SCHPC_bright_haze_minmax/tfrecords/'
            EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_age_sex_10ua_train_with_cha_no_transient_protein_process_val_with_full_new_schpc_code/data/data_w_id_bright_haze995_standardization/tfrecords/'
            
            # EXT_DIR  = '/home/latte/project/retina_ckd/code/eGFR_imageonly_with_proteinuria_val_with_full_schpc/data/SCHPC/tfrecords/'
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            # TRAIN_DATA = []
            # for i in range(5):
            #     if i != FLAGS.random_seed:
            #         TRAIN_DATA.append(DATA_DIR + 'dev{}.chunk*.tfrecord'.format(i))
            #         # TRAIN_DATA.append(EXT_DIR + 'dev{}.chunk*.tfrecord'.format(i))

            TRAIN_DATA = [DATA_DIR + 'pos*.chunk*.tfrecord']
            VALID_DATA = [DATA_DIR + 'valid*.chunk*.tfrecord']
                        #   EXT_DIR + 'dev{}.chunk*.tfrecord'.format(FLAGS.random_seed)]
            
            if FLAGS.hospital_source_type_for_prediction=='SCHPC':
              TEST_DATA  = [EXT_DIR + 'SCHPC*.chunk*.tfrecord']#['./data/SCHPC_test/' + 'test*.chunk*.tfrecord']#, EXT_DIR + 'test*.chunk*.tfrecord']
            elif FLAGS.hospital_source_type_for_prediction=='CHA' and FLAGS.data_type_for_prediction=='test':
              # TEST_DATA  = [EXT_DIR + 'test*.chunk*.tfrecord']
              TEST_DATA  = [DATA_DIR + 'test*.chunk*.tfrecord']
            elif FLAGS.hospital_source_type_for_prediction=='CHA' and FLAGS.data_type_for_prediction=='train':
              TEST_DATA  = [DATA_DIR + 'pos*.chunk*.tfrecord']        # Make prediction on train set for calculating threshold


            train_data = parse_fn(dataset(TRAIN_DATA).shuffle(FLAGS.shuffle, reshuffle_each_iteration=True))
            valid_data = parse_fn(dataset(VALID_DATA))
            test_data  = parse_fn(dataset(TEST_DATA))


            return cls(name,
                       train=train_data.batch(FLAGS.batch, drop_remainder=True).map(augment, para),
                       valid=valid_data.batch(FLAGS.batch).map(augment_valid, para),
                       test=test_data.batch(FLAGS.batch).map(augment_valid, para),
                       nclass=nclass, colors=colors,
                       height=height, width=width)

        return name, create

augment_train = lambda x: ({'image'         : augment_mirror(x['image']),
                            'age'           : x['age'],
                            'sex'           : x['sex'],
                            'blood'         : x['blood'],
                            'bilirubin'     : x['bilirubin'],
                            'urobilinogen'  : x['urobilinogen'],
                            'ketone'        : x['ketone'],
                            'protein'       : x['protein'],
                            'nitrite'       : x['nitrite'],
                            'glucose'       : x['glucose'],
                            'leucocyte'     : x['leucocyte'],
                            'ph'            : x['ph'],
                            'sg'            : x['sg']},
                           {'gt60'          : x['label']},
                            # 'diff'          : x['zeros']},
                           {'gt60'          : x['weight']})
                            # 'diff'          : x['ones']})

augment_valid = lambda x: ({'image'         : x['image'],
                            'age'           : x['age'],
                            'sex'           : x['sex'],
                            'blood'         : x['blood'],
                            'bilirubin'     : x['bilirubin'],
                            'urobilinogen'  : x['urobilinogen'],
                            'ketone'        : x['ketone'],
                            'protein'       : x['protein'],
                            'nitrite'       : x['nitrite'],
                            'glucose'       : x['glucose'],
                            'leucocyte'     : x['leucocyte'],
                            'ph'            : x['ph'],
                            'sg'            : x['sg']},
                          {'gt60'          : x['label'],
                            # 'diff'          : x['zeros']},
                            'img_id'          : x['img_id'],   # Comment this line in using SCHPC
                            'ori_image'     : x['ori_image']},
                          {'gt60'          : x['weight']})
                            # 'diff'          : x['ones']})

DATASETS = {}
DATASETS.update([DataSet.creator('fundus', augment_train, augment_valid)])
