import tensorflow as tf
import numpy as np
import cv2 as cv
import copy

from tensorflow.keras import backend as K
from scipy.ndimage.interpolation import zoom

def generate_grad_cam(model, inputs, class_name, activation_layer):

    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        this_img_id=batch[1]['img_id'].numpy()[0].decode('utf-8')
        # if batch_idx not in [14799, 17732, 19282, 21955, 22384, 24583, 26477, 28001, 29143, 30886]:
        if this_img_id not in ['228382_2.jpg','228382_1.jpg','233693_1.jpg','233693_2.jpg','237016_1.jpg','237016_2.jpg','246269_2.jpg','246269_1.jpg','249264_1.jpg','249264_2.jpg','245105_2.jpg','245105_1.jpg','251066_1.jpg','251066_2.jpg','102625_1.jpg','102625_2.jpg','106335_2.jpg','106335_1.jpg','104336_1.jpg','104336_2.jpg']:
          continue

        with tf.GradientTape() as tape:
            print('batch_idx, this_img_id',batch_idx, this_img_id)
            predictions = model([batch[0]['image'], batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']], training=False)
            grad_val = tape.gradient(predictions['gt60'], predictions['A_K'])
            # print(grad_val.numpy())

        conv_output = predictions['A_K'][0]
        grad_val = grad_val[0]

        weights = np.mean(grad_val, axis=(0, 1))

        grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            grad_cam += w * conv_output[:, :, k]

        grad_cam = grad_cam.numpy()
        grad_cam = cv.resize(grad_cam, (512, 512))

        grad_cam = np.maximum(grad_cam, 0)

        grad_cam = grad_cam / grad_cam.max()
        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(grad_cam)

    return input_images, saliency_maps_retina

def getVanillaSaliencyMap(model, inputs, class_name, activation_layer):

    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        this_img_id=batch[1]['img_id'].numpy()[0].decode('utf-8')
        if this_img_id not in ["77_1.jpg","77_2.jpg","102_1.jpg","102_2.jpg","209_2.jpg","209_1.jpg","632_1.jpg","632_2.jpg","940_2.jpg","940_1.jpg","1350_1.jpg","1350_2.jpg","1496_2.jpg","1496_1.jpg","1573_2.jpg","1573_1.jpg","2735_1.jpg","2735_2.jpg","3794_2.jpg","3794_1.jpg","4399_1.jpg","4399_2.jpg","6044_1.jpg","6044_2.jpg","6673_1.jpg","6673_2.jpg","7720_2.jpg","7720_1.jpg","8989_1.jpg","8989_2.jpg","9212_1.jpg","9212_2.jpg","9443_1.jpg","9443_2.jpg","9488_1.jpg","9488_2.jpg","9949_1.jpg","9949_2.jpg","11473_2.jpg","11473_1.jpg","14584_1.jpg","14584_2.jpg","14730_1.jpg","14730_2.jpg","15649_1.jpg","15649_2.jpg","15834_2.jpg","15834_1.jpg","16148_2.jpg","16148_1.jpg","16434_1.jpg","16434_2.jpg","16588_1.jpg","16588_2.jpg","17030_1.jpg","17030_2.jpg","18701_1.jpg","18701_2.jpg","25120_1.jpg","25120_2.jpg","26545_1.jpg","26545_2.jpg","29867_1.jpg","29867_2.jpg","33822_1.jpg","33822_2.jpg","34101_1.jpg","34101_2.jpg","42742_2.jpg","42742_1.jpg","44009_1.jpg","44009_2.jpg","45856_1.jpg","45856_2.jpg","51584_1.jpg","51584_2.jpg","51926_1.jpg","51926_2.jpg","55027_1.jpg","55027_2.jpg","58160_1.jpg","58160_2.jpg","59445_1.jpg","59445_2.jpg","61742_1.jpg","61742_2.jpg","63333_1.jpg","63333_2.jpg","63862_1.jpg","63862_2.jpg","70037_1.jpg","70037_2.jpg","71358_1.jpg","71358_2.jpg","71986_2.jpg","71986_1.jpg","72259_1.jpg","72259_2.jpg","72925_1.jpg","72925_2.jpg","73107_1.jpg","73107_2.jpg","73191_1.jpg","73191_2.jpg","73201_2.jpg","73201_1.jpg","73935_1.jpg","73935_2.jpg","74505_1.jpg","74505_2.jpg","75088_1.jpg","75088_2.jpg","75418_1.jpg","75418_2.jpg","76121_1.jpg","76121_2.jpg","76223_2.jpg","76223_1.jpg","76712_1.jpg","76712_2.jpg","77075_1.jpg","77075_2.jpg","77477_2.jpg","77477_1.jpg","78106_1.jpg","78106_2.jpg","78107_1.jpg","78107_2.jpg","78135_1.jpg","78135_2.jpg","78307_1.jpg","78307_2.jpg","78353_1.jpg","78353_2.jpg","79137_1.jpg","79137_2.jpg","79261_1.jpg","79261_2.jpg","79818_1.jpg","79818_2.jpg","80026_1.jpg","80026_2.jpg","80392_1.jpg","80392_2.jpg","80575_1.jpg","80575_2.jpg","81238_1.jpg","81238_2.jpg","81990_1.jpg","81990_2.jpg","83005_1.jpg","83005_2.jpg","83509_1.jpg","83509_2.jpg","83584_1.jpg","83584_2.jpg","84078_1.jpg","84078_2.jpg","84457_1.jpg","84457_2.jpg","85053_2.jpg","85053_1.jpg","85247_1.jpg","85247_2.jpg","85301_1.jpg","85301_2.jpg","85757_1.jpg","85757_2.jpg","87233_1.jpg","87233_2.jpg","87786_1.jpg","87786_2.jpg","88288_2.jpg","88288_1.jpg","88385_1.jpg","88385_2.jpg","88645_1.jpg","88645_2.jpg","89950_2.jpg","89950_1.jpg","90373_2.jpg","90373_1.jpg","90385_1.jpg","90385_2.jpg","90454_1.jpg","90454_2.jpg","90834_1.jpg","90834_2.jpg","91155_2.jpg","91155_1.jpg","91743_1.jpg","91743_2.jpg","300665_2.jpg","300665_1.jpg","306772_1.jpg","306772_2.jpg","309385_1.jpg","309385_2.jpg","309447_2.jpg","309447_1.jpg","310593_2.jpg","310593_3.jpg","312289_1.jpg","312289_2.jpg","313618_1.jpg","313618_2.jpg","315652_2.jpg","315652_1.jpg","318843_1.jpg","318843_2.jpg","319529_1.jpg","319529_2.jpg","319823_1.jpg","319823_2.jpg","320613_1.jpg","320613_2.jpg","221822_1.jpg","221822_2.jpg","221878_1.jpg","221878_2.jpg","222083_2.jpg","222083_1.jpg","228228_2.jpg","228228_1.jpg","228471_2.jpg","228471_1.jpg","223080_2.jpg","223080_1.jpg","226405_2.jpg","226405_1.jpg","229374_2.jpg","229374_1.jpg","223527_1.jpg","223527_2.jpg","229604_2.jpg","229604_1.jpg","230414_2.jpg","230414_1.jpg","224653_2.jpg","224653_1.jpg","227846_1.jpg","227846_2.jpg","227883_2.jpg","227883_1.jpg","230781_2.jpg","230781_1.jpg","230885_1.jpg","230885_2.jpg","241878_2.jpg","241878_1.jpg","236320_2.jpg","236320_1.jpg","242902_2.jpg","242902_1.jpg","242954_1.jpg","242954_2.jpg","233360_2.jpg","233360_1.jpg","236548_2.jpg","236548_1.jpg","236667_2.jpg","236667_1.jpg","239387_1.jpg","239387_2.jpg","233825_1.jpg","233825_2.jpg","236993_2.jpg","236993_1.jpg","234207_2.jpg","234207_1.jpg","237220_1.jpg","237220_2.jpg","240171_1.jpg","240171_2.jpg","242536_2.jpg","242536_1.jpg","237693_2.jpg","237693_1.jpg","237767_2.jpg","237767_1.jpg","240562_1.jpg","240562_2.jpg","242678_1.jpg","242678_2.jpg","234946_2.jpg","234946_1.jpg","238008_1.jpg","238008_2.jpg","240721_2.jpg","240721_1.jpg","232345_2.jpg","232345_1.jpg","232658_2.jpg","232658_1.jpg","238729_2.jpg","238729_1.jpg","238737_2.jpg","238737_1.jpg","235797_1.jpg","235797_2.jpg","235801_1.jpg","235801_2.jpg","246003_1.jpg","246003_2.jpg","248815_2.jpg","248815_1.jpg","248933_2.jpg","248933_1.jpg","251900_1.jpg","251900_2.jpg","254821_2.jpg","254821_1.jpg","254928_1.jpg","254928_2.jpg","246154_1.jpg","246154_2.jpg","255016_2.jpg","255016_1.jpg","255028_1.jpg","255028_3.jpg","252227_2.jpg","252227_1.jpg","252561_1.jpg","252561_2.jpg","246891_1.jpg","246891_2.jpg","246914_1.jpg","246914_2.jpg","249850_1.jpg","249850_2.jpg","249862_2.jpg","249862_1.jpg","247957_2.jpg","247957_1.jpg","248097_2.jpg","248097_1.jpg","254119_2.jpg","254119_1.jpg","254181_1.jpg","254181_2.jpg","248354_2.jpg","248354_1.jpg","266960_2.jpg","266960_1.jpg","263564_2.jpg","263564_1.jpg","258649_2.jpg","258649_1.jpg","261656_1.jpg","261656_2.jpg","102402_2.jpg","102402_1.jpg","102594_1.jpg","102594_2.jpg","107199_1.jpg","107199_2.jpg","107249_1.jpg","107249_2.jpg","100191_2.jpg","100191_1.jpg","100319_1.jpg","100319_2.jpg","101160_1.jpg","101160_2.jpg","106683_1.jpg","106683_2.jpg","107649_1.jpg","107649_2.jpg","107931_2.jpg","107931_1.jpg","107937_1.jpg","107937_2.jpg","107959_1.jpg","107959_2.jpg","108031_1.jpg","108031_2.jpg","106234_1.jpg","106234_2.jpg","106274_1.jpg","106274_2.jpg","106278_1.jpg","106278_2.jpg","106396_1.jpg","106396_2.jpg","109536_1.jpg","109536_2.jpg","109575_2.jpg","109575_1.jpg","109724_1.jpg","109724_2.jpg","109835_2.jpg","109835_1.jpg","110058_1.jpg","110058_2.jpg","110096_2.jpg","110096_1.jpg","110112_2.jpg","110112_1.jpg","109370_1.jpg","109370_2.jpg","109376_3.jpg","109376_4.jpg","111326_1.jpg","111326_2.jpg","111310_2.jpg","111310_1.jpg","108302_1.jpg","108302_2.jpg","108427_1.jpg","108427_2.jpg","111459_1.jpg","111459_2.jpg","108627_2.jpg","108627_1.jpg","108640_1.jpg","108640_2.jpg","108615_1.jpg","108615_2.jpg","110816_2.jpg","110816_1.jpg","110839_1.jpg","110839_2.jpg","110214_1.jpg","110214_2.jpg","110322_1.jpg","110322_2.jpg","111000_1.jpg","111000_2.jpg","109241_1.jpg","109241_3.jpg","109315_1.jpg","109315_2.jpg","103231_1.jpg","103231_2.jpg","103784_1.jpg","103784_2.jpg","104903_2.jpg","104903_1.jpg","104997_1.jpg","104997_2.jpg","105543_1.jpg","105543_2.jpg","105559_1.jpg","105559_2.jpg","105644_1.jpg","105644_2.jpg","105761_2.jpg","105761_1.jpg","322214_2.jpg","322214_1.jpg","23459_4.jpg","23459_2.jpg","37613_3.jpg","37613_2.jpg","39166_3.jpg","39166_2.jpg","63006_3.jpg","63006_2.jpg","100060_3.jpg","100060_2.jpg","104567_4.jpg","104567_3.jpg","106212_3.jpg","106212_1.jpg","106519_2.jpg","106519_1.jpg","106924_3.jpg","106924_4.jpg","106982_4.jpg","106982_3.jpg","107620_3.jpg","107620_2.jpg","107878_4.jpg","107878_3.jpg","108972_4.jpg","108972_3.jpg","109439_3.jpg","109439_5.jpg","109444_5.jpg","109444_3.jpg","110442_1.jpg","110442_3.jpg","110545_3.jpg","110545_4.jpg","110585_2.jpg","110585_1.jpg","110717_4.jpg","110717_1.jpg","111026_2.jpg","111026_5.jpg","111508_3.jpg","111508_1.jpg"]:
          continue

        with tf.GradientTape() as tape:

            # https://stackoverflow.com/questions/55066710/computing-gradients-wrt-model-inputs-in-tensorflow-eager-mode
            input_image=batch[0]['image']
            tape.watch(input_image)
            # input_image = tf.Variable(batch[0]['image'], dtype=tf.float32)

            print('batch_idx, this_img_id',batch_idx, this_img_id)
            predictions = model([input_image, batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']], training=False)
            grad_val = tape.gradient(predictions['gt60'], input_image)

        conv_output = batch[0]['image']

        dgrad_abs = tf.math.abs(grad_val)
        dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

        arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
        grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(grad_eval)

    return input_images, saliency_maps_retina

def getSmoothVanillaSaliencyMap(model, inputs, class_name, activation_layer):
    
    n_samples     = 25
    stdev_spread  = 0.15
    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        this_img_id=batch[1]['img_id'].numpy()[0].decode('utf-8')

        if this_img_id not in ["77_1.jpg","77_2.jpg","102_1.jpg","102_2.jpg","940_2.jpg","940_1.jpg","1496_2.jpg","1496_1.jpg","1573_2.jpg","1573_1.jpg","2735_1.jpg","2735_2.jpg","4399_1.jpg","4399_2.jpg","8989_1.jpg","8989_2.jpg","9212_1.jpg","9212_2.jpg","9443_1.jpg","9443_2.jpg","14584_1.jpg","14584_2.jpg","15649_1.jpg","15649_2.jpg","16434_1.jpg","16434_2.jpg","16588_1.jpg","16588_2.jpg","18701_1.jpg","18701_2.jpg","26545_1.jpg","26545_2.jpg","29867_1.jpg","29867_2.jpg","33822_1.jpg","33822_2.jpg","34101_1.jpg","34101_2.jpg","42742_2.jpg","42742_1.jpg","44009_1.jpg","44009_2.jpg","45856_1.jpg","45856_2.jpg","51584_1.jpg","51584_2.jpg","51926_1.jpg","51926_2.jpg","55027_1.jpg","55027_2.jpg","58160_1.jpg","58160_2.jpg","61742_1.jpg","61742_2.jpg","63333_1.jpg","63333_2.jpg","70037_1.jpg","70037_2.jpg","71986_2.jpg","71986_1.jpg","72259_1.jpg","72259_2.jpg","73107_1.jpg","73107_2.jpg","73201_2.jpg","73201_1.jpg","73935_1.jpg","73935_2.jpg","74505_1.jpg","74505_2.jpg","75088_1.jpg","75088_2.jpg","75418_1.jpg","75418_2.jpg","76223_2.jpg","76223_1.jpg","76712_1.jpg","76712_2.jpg","77477_2.jpg","77477_1.jpg","78307_1.jpg","78307_2.jpg","78353_1.jpg","78353_2.jpg","79137_1.jpg","79137_2.jpg","79261_1.jpg","79261_2.jpg","79818_1.jpg","79818_2.jpg","80392_1.jpg","80392_2.jpg","80575_1.jpg","80575_2.jpg","83005_1.jpg","83005_2.jpg","83584_1.jpg","83584_2.jpg","85247_1.jpg","85247_2.jpg","85301_1.jpg","85301_2.jpg","87233_1.jpg","87233_2.jpg","87786_1.jpg","87786_2.jpg","88288_2.jpg","88288_1.jpg","88385_1.jpg","88385_2.jpg","88645_1.jpg","88645_2.jpg","90385_1.jpg","90385_2.jpg","90454_1.jpg","90454_2.jpg","91155_2.jpg","91155_1.jpg","91743_1.jpg","91743_2.jpg","306772_1.jpg","306772_2.jpg","309385_1.jpg","309385_2.jpg","309447_2.jpg","309447_1.jpg","310593_2.jpg","310593_3.jpg","313618_1.jpg","313618_2.jpg","319823_1.jpg","319823_2.jpg","320613_1.jpg","320613_2.jpg","221822_1.jpg","221822_2.jpg","221878_1.jpg","221878_2.jpg","222083_2.jpg","222083_1.jpg","228228_2.jpg","228228_1.jpg","228471_2.jpg","228471_1.jpg","223080_2.jpg","223080_1.jpg","226405_2.jpg","226405_1.jpg","229374_2.jpg","229374_1.jpg","223527_1.jpg","223527_2.jpg","229604_2.jpg","229604_1.jpg","230414_2.jpg","230414_1.jpg","224653_2.jpg","224653_1.jpg","227846_1.jpg","227846_2.jpg","227883_2.jpg","227883_1.jpg","230781_2.jpg","230781_1.jpg","230885_1.jpg","230885_2.jpg","241878_2.jpg","241878_1.jpg","236320_2.jpg","236320_1.jpg","242902_2.jpg","242902_1.jpg","242954_1.jpg","242954_2.jpg","233360_2.jpg","233360_1.jpg","236548_2.jpg","236548_1.jpg","236667_2.jpg","236667_1.jpg","239387_1.jpg","239387_2.jpg","233825_1.jpg","233825_2.jpg","236993_2.jpg","236993_1.jpg","234207_2.jpg","234207_1.jpg","237220_1.jpg","237220_2.jpg","240171_1.jpg","240171_2.jpg","242536_2.jpg","242536_1.jpg","237693_2.jpg","237693_1.jpg","237767_2.jpg","237767_1.jpg","240562_1.jpg","240562_2.jpg","242678_1.jpg","242678_2.jpg","234946_2.jpg","234946_1.jpg","238008_1.jpg","238008_2.jpg","240721_2.jpg","240721_1.jpg","232345_2.jpg","232345_1.jpg","232658_2.jpg","232658_1.jpg","238729_2.jpg","238729_1.jpg","238737_2.jpg","238737_1.jpg","235797_1.jpg","235797_2.jpg","235801_1.jpg","235801_2.jpg","246003_1.jpg","246003_2.jpg","248815_2.jpg","248815_1.jpg","248933_2.jpg","248933_1.jpg","251900_1.jpg","251900_2.jpg","254821_2.jpg","254821_1.jpg","254928_1.jpg","254928_2.jpg","246154_1.jpg","246154_2.jpg","255016_2.jpg","255016_1.jpg","255028_1.jpg","255028_3.jpg","252227_2.jpg","252227_1.jpg","252561_1.jpg","252561_2.jpg","246891_1.jpg","246891_2.jpg","246914_1.jpg","246914_2.jpg","249850_1.jpg","249850_2.jpg","249862_2.jpg","249862_1.jpg","247957_2.jpg","247957_1.jpg","248097_2.jpg","248097_1.jpg","254119_2.jpg","254119_1.jpg","254181_1.jpg","254181_2.jpg","248354_2.jpg","248354_1.jpg","266960_2.jpg","266960_1.jpg","263564_2.jpg","263564_1.jpg","258649_2.jpg","258649_1.jpg","261656_1.jpg","261656_2.jpg","102402_2.jpg","102402_1.jpg","102594_1.jpg","102594_2.jpg","107199_1.jpg","107199_2.jpg","107249_1.jpg","107249_2.jpg","100191_2.jpg","100191_1.jpg","100319_1.jpg","100319_2.jpg","101160_1.jpg","101160_2.jpg","106683_1.jpg","106683_2.jpg","107649_1.jpg","107649_2.jpg","107931_2.jpg","107931_1.jpg","107937_1.jpg","107937_2.jpg","107959_1.jpg","107959_2.jpg","108031_1.jpg","108031_2.jpg","106234_1.jpg","106234_2.jpg","106274_1.jpg","106274_2.jpg","106278_1.jpg","106278_2.jpg","106396_1.jpg","106396_2.jpg","109536_1.jpg","109536_2.jpg","109575_2.jpg","109575_1.jpg","109724_1.jpg","109724_2.jpg","109835_2.jpg","109835_1.jpg","110058_1.jpg","110058_2.jpg","110096_2.jpg","110096_1.jpg","110112_2.jpg","110112_1.jpg","109370_1.jpg","109370_2.jpg","109376_3.jpg","109376_4.jpg","111326_1.jpg","111326_2.jpg","111310_2.jpg","111310_1.jpg","108302_1.jpg","108302_2.jpg","108427_1.jpg","108427_2.jpg","111459_1.jpg","111459_2.jpg","108627_2.jpg","108627_1.jpg","108640_1.jpg","108640_2.jpg","108615_1.jpg","108615_2.jpg","110816_2.jpg","110816_1.jpg","110839_1.jpg","110839_2.jpg","110214_1.jpg","110214_2.jpg","110322_1.jpg","110322_2.jpg","111000_1.jpg","111000_2.jpg","109241_1.jpg","109241_3.jpg","109315_1.jpg","109315_2.jpg","103231_1.jpg","103231_2.jpg","103784_1.jpg","103784_2.jpg","104997_1.jpg","104997_2.jpg","105559_1.jpg","105559_2.jpg","105644_1.jpg","105644_2.jpg","23459_4.jpg","23459_2.jpg","63006_3.jpg","63006_2.jpg","100060_3.jpg","100060_2.jpg","104567_4.jpg","104567_3.jpg","106212_3.jpg","106212_1.jpg","106519_2.jpg","106519_1.jpg","106924_3.jpg","106924_4.jpg","106982_4.jpg","106982_3.jpg","107620_3.jpg","107620_2.jpg","107878_4.jpg","107878_3.jpg","108972_4.jpg","108972_3.jpg","109439_3.jpg","109439_5.jpg","109444_5.jpg","109444_3.jpg","110442_1.jpg","110442_3.jpg","110545_3.jpg","110545_4.jpg","110585_2.jpg","110585_1.jpg","110717_4.jpg","110717_1.jpg","111026_2.jpg","111026_5.jpg","111508_3.jpg","111508_1.jpg"]:
          continue
        
        stdev = stdev_spread / (batch[0]['image'].numpy().max() - batch[0]['image'].numpy().min())
        std_tensor = np.ones_like(batch[0]['image']) * stdev
        
        print('batch_idx, this_img_id',batch_idx, this_img_id)
        
        for i in range(n_samples):
        
          with tf.GradientTape() as tape:

              # https://stackoverflow.com/questions/55066710/computing-gradients-wrt-model-inputs-in-tensorflow-eager-mode
              input_image=batch[0]['image']

              # Noised input image
              noised_input_image=copy.deepcopy(input_image)+tf.random.normal(batch[0]['image'].shape, mean=0, stddev=stdev)

              tape.watch(noised_input_image)

              predictions = model([noised_input_image, batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']], training=False)

              grad_val = tape.gradient(predictions['gt60'], noised_input_image)

          conv_output = batch[0]['image']

          dgrad_abs = tf.math.abs(grad_val)
          dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
          arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
          grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

          if i == 0:
              total_cams = grad_eval.copy()
          else:
              total_cams += grad_eval

        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(total_cams/n_samples)

    return input_images, saliency_maps_retina


def getSmoothGradCAMPP(model, inputs, class_name, activation_layer):
    """
    Args:
        x: input image. shape =>(1, 3, H, W)
    Return:
        Total_cams: mean of class activation mappings of n_samples
    """
    n_samples     = 25
    stdev_spread  = 0.15
    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        this_img_id=batch[1]['img_id'].numpy()[0].decode('utf-8')
        
        if this_img_id not in ['228382_2.jpg','228382_1.jpg','233693_1.jpg','233693_2.jpg','237016_1.jpg','237016_2.jpg','246269_2.jpg','246269_1.jpg','249264_1.jpg','249264_2.jpg','245105_2.jpg','245105_1.jpg','251066_1.jpg','251066_2.jpg','102625_1.jpg','102625_2.jpg','106335_2.jpg','106335_1.jpg','104336_1.jpg','104336_2.jpg']:
          continue

        stdev = stdev_spread / (batch[0]['image'].numpy().max() - batch[0]['image'].numpy().min())
        std_tensor = np.ones_like(batch[0]['image']) * stdev
        
        print('batch_idx, this_img_id',batch_idx, this_img_id)
        
        for i in range(n_samples):
            # Noised input image
            image = batch[0]['image'] + tf.random.normal(batch[0]['image'].shape, mean=0, stddev=stdev)
            
            with tf.GradientTape() as tape:

                # Forward pass with noised input image
                predictions = model([batch[0]['image'], batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], 
                                     batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], 
                                     batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']], training=False)

                # Gradient of class score with respect to feature map (from noised input)
                grad_val = tape.gradient(predictions['gt60'], predictions['A_K'])

            # feature map (from noised input)
            conv_layer_output = predictions['A_K']
            
            # Gradients from noised input
            conv_first_grad  = tf.math.exp(predictions['gt60']) * grad_val
            conv_second_grad = tf.math.exp(predictions['gt60']) * grad_val * grad_val
            conv_third_grad  = tf.math.exp(predictions['gt60']) * grad_val * grad_val * grad_val

            global_sum = np.sum(conv_layer_output[0].numpy().reshape((-1,conv_first_grad[0].shape[2])), axis=0)
            
            alpha_num   = conv_second_grad[0]
            alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
            alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
            alphas = alpha_num/alpha_denom

            weights = np.maximum(conv_first_grad[0], 0.0)
            
            alphas_thresholding = np.where(weights, alphas, 0.0)

            alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
            alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))

            alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))
            
            deep_linearization_weights = np.sum((weights * alphas).numpy().reshape((-1,conv_first_grad[0].shape[2])),axis=0)
            
            grad_CAM_map = np.sum(deep_linearization_weights*conv_layer_output[0], axis=2)

            cam = np.maximum(grad_CAM_map, 0)
            cam = zoom(cam,512/cam.shape[0])
            smg_cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) # scale 0 to 1.0

            if i == 0:
                total_cams = smg_cam.copy()
            else:
                total_cams += smg_cam

        total_cams /= n_samples
        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(total_cams)
    
    return input_images, saliency_maps_retina


def getGradCAMPP(model, inputs, class_name, target_layer):
    
    input_images, saliency_maps_retina = [], []
    for (batch_idx, (batch)) in enumerate(inputs):
        print(batch_idx)
        if batch_idx==10:
          break
        with tf.GradientTape() as tape:
            predictions = model([batch[0]['image'], batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']], training=False)
            grad_val = tape.gradient(predictions['gt60'], predictions['A_K'])

        conv_layer_output = predictions['A_K']
        conv_first_grad  = tf.math.exp(predictions['gt60']) * grad_val
        conv_second_grad = tf.math.exp(predictions['gt60']) * grad_val * grad_val
        conv_third_grad  = tf.math.exp(predictions['gt60']) * grad_val * grad_val * grad_val

        global_sum = np.sum(conv_layer_output[0].numpy().reshape((-1,conv_first_grad[0].shape[2])), axis=0)
        
        alpha_num   = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        
        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))

        alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))
        
        deep_linearization_weights = np.sum((weights * alphas).numpy().reshape((-1,conv_first_grad[0].shape[2])),axis=0)
        
        grad_CAM_map = np.sum(deep_linearization_weights*conv_layer_output[0], axis=2)

        if batch_idx==2:
          print('grad_CAM_map',grad_CAM_map)

        cam = np.maximum(grad_CAM_map, 0)
        cam = zoom(cam,512/cam.shape[0])
        
        cam = cam / (np.max(cam)) # scale 0 to 1.0
        input_images.append(batch[1]['ori_image'][0].numpy())
        saliency_maps_retina.append(cam)
    
    return input_images, saliency_maps_retina