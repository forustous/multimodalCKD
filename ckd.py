import pandas as pd
import numpy as np
import tensorflow as tf

import cv2 as cv
import os
from PIL import Image

import skimage.filters.rank as sfr
from skimage.morphology import square

PATH = "./"

ckd_model = tf.keras.models.load_model('./ckd_model.h5py')
# ckd_model.summary()

V_MEAN_MEAN, V_MEAN_STD = 2.9393, 6.9289
outlier_upper_bound_vs=241.8319454672273
outlier_lower_bound_vs=41.050601426582006
haze_outlier_upper_bount=0.5295
cropped_size = 512


def run(file):
    image = file.copy()
    image = np.array(image)
    if image is not None:
        output = image.copy()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        if gray.shape == (1296, 1920):
            x, y, z = 944, 940, 721
        elif gray.shape == (3000, 4496):
            x, y, z = 2228, 2207, 1480
        elif gray.shape == (3264, 4928):
            x, y, z = 2461, 2459, 1561
        elif gray.shape == (4496, 3000):
            x, y, z = 0, 0, 0 # Problem
            print("4496, 3000")
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
            print("614, 816")
            pass
        elif gray.shape == (720, 960):
            x, y, z = 0, 0, 0
            print("720, 960")
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
            print("This image is an unsupported size.")
            pass

        if width > height:
            num_borders = (width - height) // 2
            rep_image = cv.copyMakeBorder(output, num_borders, num_borders, 0, 0, cv.BORDER_REFLECT)
        else:
            num_borders = (height - width) // 2
            rep_image = cv.copyMakeBorder(output, 0, 0, num_borders, num_borders, cv.BORDER_REFLECT)

        output = rep_image.copy()
        gray = cv.cvtColor(rep_image, cv.COLOR_BGR2GRAY)
        mask_height, mask_width = gray.shape

        mask = np.zeros((mask_height, mask_width), np.uint8)
        cv.circle(mask, (x, y), z, (255, 255, 255), -1)


        crop_CN, cropped_mask, crop = crop_images(output, mask)

        hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        split_size = int(cropped_size/2)
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
            # cv.imwrite(os.path.join(ODD_PATH, 'outlier_' + file_name), self.image)
            print("This image is an outlier.")

        if x == 0:
            crop_CN, cropped_mask = None, None

        if crop_CN is not None :
            avg = haze_scoring(crop)
            Mh = 300000 # 상수
            haze_score = '%.3f' % round(Mh / avg, 4) # haze_score

            if float(haze_score) >= haze_outlier_upper_bount :
                crop_CN, cropped_mask = None, None
                haze = True

        return crop_CN

def haze_scoring(crop) :
    crop_reitna = crop[83:429, 83:429].copy() # retina 영역
    src = cv.resize(crop_reitna, dsize=(512,512), interpolation=cv.INTER_AREA) # resize
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 이미지 이진화
    reverse = cv.bitwise_not(src_gray) # 이미지 반전
    low_kernel = np.ones((32,32), np.float32)/(32*32) # 32 * 32 kernel 생성
    low_img = cv.filter2D(reverse, -1, low_kernel) # lowpass filter
    high_img = reverse - low_img # highpass filter
    ent_img = sfr.entropy(high_img, square(3)) # local entropy 적용
    f = np.fft.fft2(ent_img) # 퓨리에 변환
    f_shift = np.fft.fftshift(f)
    abs_fouier_transform = np.abs(f_shift)
    power_spectrum = np.square(abs_fouier_transform) # 파워 스펙트럼
    flat = power_spectrum.flatten()
    sum = 0
    for ind in range(0, len(flat)) :
        sum = sum + flat[ind]
    avg = sum/len(flat)

    return avg

def crop_images(output, mask):
    masked_data = cv.bitwise_and(output, output, mask=mask)
    _, thresh = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

    countours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(countours[0][0])

    crop = masked_data[y:y + h, x:x + w, :]
    crop = cv.resize(crop, dsize=(cropped_size, cropped_size), interpolation=cv.INTER_AREA)

    cropped_mask = mask[y:y + h, x:x + w]
    cropped_mask = cv.resize(cropped_mask, dsize=(cropped_size, cropped_size), interpolation=cv.INTER_AREA)

    clahe_img = clahe(crop)
    color_n_img = color_normalization(clahe_img, cropped_mask)

    return color_n_img, cropped_mask, crop

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
                    128 # 결과 영상에 추가적으로 더할 값
                )

    b = a * np.stack([mask, mask, mask], axis=2)
    b = np.asarray(b, dtype='uint8')
    
    return b

def predict_model(cropped_image, age, sex, blood, bilirubin, urobilinogen, ketone, protein, nitrite, glucose, leucocyte, ph, sg, retina_model) :
    image = (cropped_image[:, :, ::-1])
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = image - mean
    stddev = np.sqrt(np.mean(image ** 2, axis=(0, 1), keepdims=True))
    image = image / stddev

    age       = (float(age) - 45.4695) / 11.7626
    sex       = float(sex)

    blood       = (float(blood) - 0.4222) / 0.8937
    bilirubin       = (float(bilirubin) - 0.03446) / 0.2683
    urobilinogen       = (float(urobilinogen) - 0.07676) / 0.4077
    ketone       = (float(ketone) - 0.1999) / 0.6776
    protein       = (float(protein) - 0.2496) / 0.5990
    nitrite       = (float(nitrite) - 0.00912) / 0.09506
    glucose       = (float(glucose) - 0.1507) / 0.7654
    leucocyte       = (float(leucocyte) - 0.31601) / 0.8468
    ph       = (float(ph) - 6.1786) / 0.7942
    sg       = (float(sg) - 1.0187) / 0.0089


    i1 = np.asarray([image])
    i2 = np.asarray([[age]])
    i3 = np.asarray([[sex]])
    i4 = np.asarray([[blood]])
    i5 = np.asarray([[bilirubin]])
    i6 = np.asarray([[urobilinogen]])
    i7 = np.asarray([[ketone]])
    i8 = np.asarray([[protein]])
    i9 = np.asarray([[nitrite]])
    i10 = np.asarray([[glucose]])
    i11 = np.asarray([[leucocyte]])
    i12 = np.asarray([[ph]])
    i13 = np.asarray([[sg]])

    ckd_pred = ckd_model([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13])['gt60'].numpy()[0][0]

    return ckd_pred

def main():
    try:
        image = Image.open(os.path.join(PATH, 'image.jpg'))
    except FileNotFoundError:
        print("Error: Image file not found.")
        return None

    try:
        df = pd.read_csv(os.path.join(PATH, 'urine.csv'))
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None

    age = df['age'].iloc[0] if not df['age'].empty else 0
    sex = df['sex'].iloc[0] if not df['sex'].empty else 0
    blood = df['blood'].iloc[0] if not df['blood'].empty else 0
    bilirubin = df['bilirubin'].iloc[0] if not df['bilirubin'].empty else 0
    urobilinogen = df['urobilinogen'].iloc[0] if not df['urobilinogen'].empty else 0
    ketone = df['ketone'].iloc[0] if not df['ketone'].empty else 0
    protein = df['protein'].iloc[0] if not df['protein'].empty else 0
    nitrite = df['nitrite'].iloc[0] if not df['nitrite'].empty else 0
    glucose = df['glucose'].iloc[0] if not df['glucose'].empty else 0
    leucocyte = df['leucocyte'].iloc[0] if not df['leucocyte'].empty else 0
    ph = df['ph'].iloc[0] if not df['ph'].empty else 0
    sg = df['sg'].iloc[0] if not df['sg'].empty else 0

    if sex == 'M' :
        sex = 1
    elif sex == 'F' :
        sex = 0

    cropped_image = run(image)
    prediction = predict_model(cropped_image, age, sex, blood, bilirubin, urobilinogen, ketone, protein, nitrite, glucose, leucocyte, ph, sg, ckd_model)

    print(f"The probability of CKD is {prediction:.3f}.")

main()