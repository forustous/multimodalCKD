import numpy as np
import pandas as pd
import tensorflow as tf
import skimage

import cv2 as cv
import os
import shutil
import pickle
import gzip

from absl import flags, app
from tqdm import tqdm, trange
from multiprocessing import Pool
from utils import get_logger
from skimage.morphology import disk
from skimage.filters import rank
import skimage.filters.rank as sfr
from skimage.morphology import square

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = flags.FLAGS

# DATA_PATH   = '/home/dclab/hard/workspace/Fundus/data/devset_cha'#'./data/images_one'
DATA_PATH   = '/home/latte/project/retina_ckd/data/image/devset_cha'#'./data/images_one'
# OUTPUT_PATH = './data/devset'
# OUTPUT_PATH = './data_w_id/devset'
OUTPUT_PATH = './data_w_id_bright_haze_minmax/devset'
ODD_PATH    = DATA_PATH + '_odd'
HAZE_PATH = DATA_PATH + '_haze'
# NUM_WORKERS = 16
NUM_WORKERS = 86

V_MEAN_MEAN, V_MEAN_STD = 2.9393, 6.9289
outlier_upper_bound_vs=241.8319454672273
outlier_lower_bound_vs=41.050601426582006
haze_outlier_upper_bount=0.5295

class preprocess():
    def __init__(self, data_path=DATA_PATH, cropped_size=512):
        self.DATA_PATH = data_path
        self.cropped_size = cropped_size

    def run(self, file_name):
        self.image = cv.imread(os.path.join(self.DATA_PATH, file_name))
        if self.image is not None:
            output = self.image.copy()
            gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            self.height, self.width = gray.shape
            if gray.shape == (1296, 1920):
                x, y, z = 944, 940, 721
            elif gray.shape == (3000, 4496):
                x, y, z = 2228, 2207, 1480
            elif gray.shape == (3264, 4928):
                x, y, z = 2461, 2459, 1561
            elif gray.shape == (4496, 3000):
                x, y, z = 0, 0, 0 # Problem
                print("4496, 3000", file_name)
            elif gray.shape == (2592, 3872):
                # print(file_name)
                x, y, z = 1780, 1950, 1450
            elif gray.shape == (2848, 4288):
                x, y, z = 1985, 2135, 1583
            elif gray.shape == (2448, 3696):
                x, y, z = 1806, 1832, 1367
            elif gray.shape == (1296, 1920):
                x, y, z = 960, 937, 708
            elif gray.shape == (1296, 1936):
                x, y, z = 960, 950, 720
            elif gray.shape == (1632, 2464):
                x, y, z = 1220, 1250, 790
            elif gray.shape == (954, 1440):
                x, y, z = 700, 720, 530
            elif gray.shape == (936, 1440):
                x, y, z = 720, 720, 540
            elif gray.shape == (956, 1440):
                x, y, z = 720, 720, 530
            elif gray.shape == (614, 816):
                x, y, z = 0, 0, 0
                print("614, 816", file_name)
                pass
            elif gray.shape == (720, 960):
                x, y, z = 0, 0, 0
                print("720, 960", file_name)
                pass
            elif gray.shape == (2000, 2992):
                x, y, z = 1500, 1500, 933
            elif gray.shape == (2136, 3216):
                x, y, z = 1600, 1600, 1030
            elif gray.shape == (4000, 6000):
                x, y, z = 3000, 3000, 1900
            else:
                print(gray.shape)
                dp = 1.0
                param1, param2 = 100, 30
                x, y, z = 0, 0, 0
                pass

            if self.width > self.height:
                num_borders = (self.width - self.height) // 2
                rep_image = cv.copyMakeBorder(output, num_borders, num_borders, 0, 0, cv.BORDER_REFLECT)
            else:
                num_borders = (self.height - self.width) // 2
                rep_image = cv.copyMakeBorder(output, 0, 0, num_borders, num_borders, cv.BORDER_REFLECT)

            output = rep_image.copy()
            gray = cv.cvtColor(rep_image, cv.COLOR_BGR2GRAY)
            mask_height, mask_width = gray.shape

            mask = np.zeros((mask_height, mask_width), np.uint8)
            if z <= 0:
                print("err", file_name)
            cv.circle(mask, (x, y), z, (255, 255, 255), -1)


            crop_CN, cropped_mask, crop = self.crop_images(output, mask)

            hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            split_size = int(self.cropped_size/2)
            h, s, v0, v1, v2, v3\
                = np.sum(h, axis=(0, 1)) / (np.sum(cropped_mask) / 255), \
                  np.sum(s, axis=(0, 1)) / (np.sum(cropped_mask) / 255), \
                  np.sum(v[:split_size, :split_size], axis=(0, 1)) / (np.sum(cropped_mask[:split_size, :split_size]) / 255), \
                  np.sum(v[:split_size, split_size:], axis=(0, 1)) / (np.sum(cropped_mask[:split_size, split_size:]) / 255), \
                  np.sum(v[split_size:, :split_size], axis=(0, 1)) / (np.sum(cropped_mask[split_size:, :split_size]) / 255), \
                  np.sum(v[split_size:, split_size:], axis=(0, 1)) / (np.sum(cropped_mask[split_size:, split_size:]) / 255)

            if (outlier_lower_bound_vs>=v0 or outlier_upper_bound_vs<=v0) or\
               (outlier_lower_bound_vs>=v1 or outlier_upper_bound_vs<=v1) or\
               (outlier_lower_bound_vs>=v2 or outlier_upper_bound_vs<=v2) or\
               (outlier_lower_bound_vs>=v3 or outlier_upper_bound_vs<=v3):
              crop_CN, cropped_mask = None, None
              cv.imwrite(os.path.join(ODD_PATH, 'outlier_' + file_name), self.image)

            if x == 0:
                crop_CN, cropped_mask = None, None

            if crop_CN is not None :
                avg = self.haze_scoring(crop)
                Mh = 300000 # constant
                haze_score = '%.3f' % round(Mh / avg, 4) # haze_score

                if float(haze_score) >= haze_outlier_upper_bount :
                    crop_CN, cropped_mask = None, None
                    cv.imwrite(os.path.join(HAZE_PATH, 'haze_' + file_name), self.image)
                    haze = True

            return crop_CN, crop, h, s, v

    def haze_scoring(self, crop) :
        crop_reitna = crop[83:429, 83:429].copy() # retina area
        src = cv.resize(crop_reitna, dsize=(512,512), interpolation=cv.INTER_AREA) # resize
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # image binarization
        reverse = cv.bitwise_not(src_gray) # image reversal
        low_kernel = np.ones((32,32), np.float32)/(32*32) # generation of 32 * 32 kernel 
        low_img = cv.filter2D(reverse, -1, low_kernel) # lowpass filter
        high_img = reverse - low_img # highpass filter
        ent_img = sfr.entropy(high_img, square(3)) # apply local entropy
        f = np.fft.fft2(ent_img) # fourier transform
        f_shift = np.fft.fftshift(f)
        abs_fouier_transform = np.abs(f_shift)
        power_spectrum = np.square(abs_fouier_transform) # power spectrum
        flat = power_spectrum.flatten()
        sum = 0
        for ind in range(0, len(flat)) :
            sum = sum + flat[ind]
        avg = sum/len(flat)

        return avg

    def crop_images(self, output, mask):
        masked_data = cv.bitwise_and(output, output, mask=mask)
        _, thresh = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

        countours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(countours[0][0])

        crop = masked_data[y:y + h, x:x + w, :]
        crop = cv.resize(crop, dsize=(self.cropped_size, self.cropped_size), interpolation=cv.INTER_AREA)

        cropped_mask = mask[y:y + h, x:x + w]
        cropped_mask = cv.resize(cropped_mask, dsize=(self.cropped_size, self.cropped_size), interpolation=cv.INTER_AREA)

        clahe_img = clahe(crop)
        color_n_img = color_normalization(clahe_img, cropped_mask)

        return color_n_img, cropped_mask, crop


def processing(df, image_list, mode, chunk_size):
    pool = Pool(NUM_WORKERS)
    n_images = len(image_list)
    preprocessor = preprocess()
    np.random.shuffle(image_list)
    chunk_offsets = _split_data(chunk_size, n_images)

    try:
        NUM_DATA = pool.map_async(_parse_data, [(cidx, begin, end, image_list, preprocessor, df, mode)
                                                for cidx, (begin, end) in enumerate(chunk_offsets)]).get(999999999)

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    num_data, eGFRs = 0, 0
    for num_images, eGFR in NUM_DATA:
        num_data += num_images
        eGFRs += eGFR

    get_logger('data').info('size of {} set: {}'.format(mode, num_data))
    get_logger('data').info('size of eGFR: {}'.format(eGFRs))

def _split_data(chunk_size, total):
    chunks = [(i, min(i + chunk_size, total))
              for i in range(0, total, chunk_size)]

    return chunks

def _parse_data(data):
    cidx, begin, end, image_list, preprocessor, df, data_type = data
    hue, saturation, values = [], [], []
    save_file_names = []
    with tqdm(total=end-begin) as pbar:
        outputfiles = os.path.join(OUTPUT_PATH, 'tfrecords', '{}.chunk{}'.format(data_type, cidx) + '.tfrecord')
        writer = tf.io.TFRecordWriter(outputfiles)
        num_eGFR = 0
        for file_name in image_list[begin:end]:
            cropped_image, cropped_mask, h, s, v = preprocessor.run(file_name)

            if cropped_image is not None:
                this_df = df[df['image_id'] == file_name]
                img_id       = str(this_df['image_id'].values[0])
                sex          = int(this_df['male'])
                age          = int(this_df['age'])
                blood        = int(this_df['blood'])
                bilirubin    = int(this_df['bilirubin'])
                urobilinogen = int(this_df['urobilinogen'])
                ketone       = int(this_df['ketone'])
                protein      = int(this_df['protein'])
                nitrite      = int(this_df['nitrite'])
                glucose      = int(this_df['glucose'])
                ph           = float(this_df['ph'])
                sg           = float(this_df['sg'])
                leucocyte    = int(this_df['leucocyte'])
                eGFR         = int(this_df['egfr_60'])
                eGFR45       = int(this_df['egfr_45'])
                CKD          = int(this_df['ckd'])

                direction = 0 if np.asarray(this_df['side'])[0] == 'L' else 1 #int(this_df['direction'])#


                num_eGFR += 1
                feat = {'image'         : _bytes_feature(cropped_image.reshape([-1]).tobytes()),
                        'mask'          : _bytes_feature(cropped_mask.reshape([-1]).tobytes()),
                        'img_id'        : _bytes_feature(img_id.encode('utf-8')),
                        'sex'           : _int64_feature(np.asarray(sex).astype(np.int64)),
                        'eGFR'          : _int64_feature(np.asarray(eGFR).astype(np.int64)),
                        'eGFR45'        : _int64_feature(np.asarray(eGFR45).astype(np.int64)),
                        'CKD'           : _int64_feature(np.asarray(CKD).astype(np.int64)),
                        'age'           : _int64_feature(np.asarray(age).astype(np.int64)),
                        'blood'         : _int64_feature(np.asarray(blood).astype(np.int64)),
                        'bilirubin'     : _int64_feature(np.asarray(bilirubin).astype(np.int64)),
                        'urobilinogen'  : _int64_feature(np.asarray(urobilinogen).astype(np.int64)),
                        'ketone'        : _int64_feature(np.asarray(ketone).astype(np.int64)),
                        'protein'       : _int64_feature(np.asarray(protein).astype(np.int64)),
                        'nitrite'       : _int64_feature(np.asarray(nitrite).astype(np.int64)),
                        'glucose'       : _int64_feature(np.asarray(glucose).astype(np.int64)),
                        'leucocyte'     : _int64_feature(np.asarray(leucocyte).astype(np.int64)),
                        'ph'            : _float_feature(np.asarray(ph).astype(np.float32)),
                        'sg'            : _float_feature(np.asarray(sg).astype(np.float32)),
                        'direction'     : _int64_feature(np.asarray(direction).astype(np.int64))
                        }

                record = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(record.SerializeToString())
                save_file_names.append(file_name)
                hue.append(h)
                saturation.append(s)
                values.append(v)
            pbar.update(1)
    writer.close()

    filename = os.path.join(OUTPUT_PATH, 'tfrecords', '{}_image_names{}.txt'.format(data_type, cidx))
    with open(filename, 'w') as f:
        for item in save_file_names:
            f.write("%s\n" % item)

    hsv_dict = {'hue': hue, 'saturation': saturation, 'values': values}

    filename = os.path.join(OUTPUT_PATH, 'tfrecords', 'hsv.chunk{}.pickle'.format(cidx))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(hsv_dict, f)

    return len(save_file_names), num_eGFR
    # _save_as_tfrecord(data_dict, 'train.chunk{}'.format(cidx))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def clahe(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    img0 = clahe.apply(img[:, :, 0])
    img1 = clahe.apply(img[:, :, 1])
    img2 = clahe.apply(img[:, :, 2])
    
    return np.stack([img0, img1, img2], axis=2)

def color_normalization(img, mask):
    scale = 200
    mask = np.asarray(mask/255, dtype='int')
    a=cv.addWeighted(img, # scaled image
                    4, # weight for a image
                    cv.GaussianBlur(img,(0,0),scale/30), # Gaussian blur on a image
                    -4, # weight for b image
                    128 # The value which is added to result image
                )

    b = a * np.stack([mask, mask, mask], axis=2)
    b = np.asarray(b, dtype='uint8')
    
    return b

def main(argv):
    del argv

    if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)
    if os.path.exists(ODD_PATH): shutil.rmtree(ODD_PATH)
    if os.path.exists(os.path.join(OUTPUT_PATH, 'tfrecords')):
        shutil.rmtree(os.path.join(OUTPUT_PATH, 'tfrecords'))

    os.mkdir(ODD_PATH)
    os.mkdir(os.path.join(OUTPUT_PATH, 'tfrecords'))

    np.random.seed(0)
    # df = pd.read_csv('./data/cha_egfr_dataset_no_transient.csv') #tbl_prog_direction.csv #tbl_binocular_dev.csv
    df = pd.read_csv('/home/latte/project/retina_ckd/data/table/cha/cha_egfr_dataset_with_proteinuria.csv') #tbl_prog_direction.csv #tbl_binocular_dev.csv
    df = df[df['image_taken'] != 'y_dmc']

    dev_data = df[df['test']==0]
    test_data = df[df['test']==1]

    dev_patients = dev_data['patient_id'].unique()
    np.random.shuffle(dev_patients)

    # Training
    begins = 0
    ends   = int(0.8 * len(dev_patients))
    train_patients = dev_patients[begins:ends]
    train_data = df[df['patient_id'].isin(train_patients)]
    processing(train_data, train_data['image_id'].unique(), 'pos', chunk_size=(len(train_data) // NUM_WORKERS) + 1)

    # Validation
    valid_patients = dev_patients[ends:]
    valid_data = df[df['patient_id'].isin(valid_patients)]
    processing(valid_data, valid_data['image_id'].unique(), 'valid', chunk_size=(len(valid_data) // NUM_WORKERS) + 1)

    # Test
    processing(test_data, test_data['image_id'].unique(), 'test', chunk_size=(len(test_data) // NUM_WORKERS) + 1)

if __name__ == '__main__':
    flags.DEFINE_integer('random_seed', 0, 'Seed')

    app.run(main)
