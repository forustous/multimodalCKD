import tensorflow as tf
import numpy as np
import cv2 as cv
import copy

from tensorflow.keras import backend as K
from scipy.ndimage.interpolation import zoom

def getVanillaSaliencyMap(model, inputs, class_name, activation_layer):

    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        this_img_id=batch[1]['img_id'].numpy()[0].decode('utf-8')

        with tf.GradientTape() as tape:

            input_image=batch[0]['image']
            tape.watch(input_image)

            print('batch_idx, this_img_id',batch_idx, this_img_id)
            predictions = model([input_image], training=False)
            grad_val = tape.gradient(predictions['gt60'], input_image)

        conv_output = batch[0]['image']

        dgrad_abs = tf.math.abs(grad_val)
        dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
        arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
        grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(grad_eval)

    return input_images, saliency_maps_retina
