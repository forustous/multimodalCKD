import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import warnings, os
import math

from tqdm import tqdm
from tensorflow.keras import backend as K
from absl import flags, app
from libml import models, utils
from libml.data import DATASETS
from libml.grad_cam import getVanillaSaliencyMap
from libml.utils import ExponentialMovingAverage

from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

FLAGS = flags.FLAGS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


class Fundus(models.MultiMoel):
    def train(self, dataset, batch, filters, repeat, scales, n_classes, test_case_name):

        self.dataset = dataset
        NUM_GPU = len(utils.get_available_gpus())
        print("Num of GPUs: {}".format(NUM_GPU))

        save_dir_name = test_case_name
        
        checkpoint_prefix = os.path.join(os.path.join('./model', save_dir_name), "best_checkpoint")


        train_data = self.dataset.train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        valid_data = self.dataset.valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


        opt = tfa.optimizers.AdamW(weight_decay=FLAGS.wd, learning_rate=FLAGS.lr)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        train_auc           = tf.keras.metrics.AUC()
        valid_auc           = tf.keras.metrics.AUC()

        history = dict({'train': [],
                        'valid': []})

        best_metric, best_epoch = 0, 0



        if NUM_GPU == 1:
            model = self.multi_model(filters=filters, n_classes=n_classes)
            model.summary()

            def train_step(batch, model):
                with tf.GradientTape() as tape:

                    predictions = model([batch[0]['image'], batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']])
                    loss = bce(batch[1]['gt60'], predictions['gt60'], sample_weight=batch[2]['gt60'])

                    grads = tape.gradient(tf.reduce_mean(loss), model.trainable_variables)
                    opt.apply_gradients(zip(grads, model.trainable_variables))

                train_auc.update_state(batch[1]['gt60'], predictions['gt60'])#[:, 1])

                return tf.reduce_mean(loss)

            def infer_step(batch, model):
                predictions = model([batch[0]['image'], batch[0]['age'], batch[0]['sex'], batch[0]['blood'], batch[0]['bilirubin'], batch[0]['urobilinogen'], batch[0]['ketone'], batch[0]['protein'], batch[0]['nitrite'], batch[0]['glucose'], batch[0]['leucocyte'], batch[0]['ph'], batch[0]['sg']])
                loss = bce(batch[1]['gt60'], predictions['gt60'], sample_weight=batch[2]['gt60']) 
                valid_auc.update_state(batch[1]['gt60'], predictions['gt60'])#[:, 1])
                
                return tf.reduce_mean(loss)

            for epoch in range(FLAGS.epochs):
                start = time.time()
                train_losses = []
                for (batch_idx, (batch)) in enumerate(train_data):
                    loss = train_step(batch, model)
                    train_losses.append(loss)

                history['train'].append(np.mean(train_losses))
                print('\n [*] Epoch {:02d}'.format(epoch+1))
                print(' [*] train_loss {:.4f} train_loss(AL): {:.4f}, train_auc {:.4f}'.format(np.mean(train_losses), np.mean(train_losses_AL), train_auc.result()))

                # Validation step
                valid_losses = []
                for (batch_idx, (batch)) in enumerate(valid_data):
                    loss = infer_step(batch, model)
                    valid_losses.append(loss)
                history['valid'].append(np.mean(valid_losses))

                this_loss = valid_auc.result()#np.mean(valid_losses)
                if this_loss > best_loss:
                    best_epoch = epoch+1
                    best_loss = this_loss
                    model.save_weights(checkpoint_prefix)
                print(' [*] valid_loss {:.4f} | valid_auc {:.4f} | Best epoch: {:02d}'.format(np.mean(valid_losses), valid_auc.result(), best_epoch))
                print(' [*] Time {:.2f}sec per epoch'.format(time.time() - start))

        #TODO: mutli-gpu setting
        else:
            @tf.function
            def train_step(batch, model):
                def step_fn(inputs, model):
                    with tf.GradientTape() as tape:
                        predictions = model([inputs[0]['image']], training=True)
                        
                        loss = bce(inputs[1]['gt60'], predictions['gt60'], sample_weight=inputs[2]['gt60'])

                        grads = tape.gradient(tf.reduce_mean(loss), model.trainable_variables)
                        opt.apply_gradients(zip(grads, model.trainable_variables))
                        train_auc.update_state(inputs[1]['gt60'], predictions['gt60'])
                    return loss

                per_example_losses = mirrored_strategy.run(step_fn, args=(batch, model))
                mean_CE_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

                return mean_CE_loss

            @tf.function
            def infer_step(batch, model):
                def step_fn(inputs, model):
                    predictions = model([inputs[0]['image']], training=False)
                    loss = bce(inputs[1]['gt60'], predictions['gt60'], sample_weight=inputs[2]['gt60'])
                    valid_auc.update_state(inputs[1]['gt60'], predictions['gt60'])#[:, 1])

                    return loss

                per_example_losses = mirrored_strategy.run(step_fn, args=(batch, model))
                mean_CE_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

                return mean_CE_loss

            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = self.multi_model(filters=filters, n_classes=n_classes)
                model.summary()
                dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_data)
                dist_valid_dataset = mirrored_strategy.experimental_distribute_dataset(valid_data)
                train_loss_list=[]
                valid_loss_list=[]
                train_auc_list=[]
                valid_auc_list=[]
                for epoch in range(FLAGS.epochs):
                    f = open("./logs/{}.out".format(save_dir_name), 'a')
                    start = time.time()
                    train_losses = []
                    for (batch_idx, (batch)) in enumerate(dist_train_dataset):
                        loss = train_step(batch, model)
                        train_losses.append(loss)

                    history['train'].append(np.mean(train_losses))
                    print('\n [*] Epoch {:02d} / {:d}'.format(epoch+1, FLAGS.epochs))
                    print(' [*] train_loss {:.4f} train_auc {:.4f}'.format(np.mean(train_losses), train_auc.result()))
                    f.write('\n\n [*] Epoch {:02d} / {:d}'.format(epoch+1, FLAGS.epochs))
                    f.write('\n [*] train_loss {:.4f} train_auc {:.4f}'.format(np.mean(train_losses), train_auc.result()))

                    # Validation step
                    valid_losses = []
                    for (batch_idx, (batch)) in enumerate(dist_valid_dataset):
                        loss = infer_step(batch, model)
                        valid_losses.append(loss)
                    history['valid'].append(np.mean(valid_losses))

                    this_metric = valid_auc.result()#np.mean(valid_losses)
                    if this_metric > best_metric:
                        best_epoch = epoch+1
                        best_metric = this_metric
                        model.save_weights(checkpoint_prefix)

                    ggg_s=checkpoint_prefix.split('/')
                    ggg_s.insert(3,str(epoch+1).zfill(3))
                    ggg_s_j="/".join(ggg_s)
                    model.save_weights(ggg_s_j)

                    print(' [*] valid_loss {:.4f} | valid_auc {:.4f} | Best epoch: {:02d}'.format(np.mean(valid_losses), valid_auc.result(), best_epoch))
                    print(' [*] Time {:.2f}sec per epoch'.format(time.time() - start))
                    f.write('\n [*] valid_loss {:.4f} | valid_auc {:.4f} | Best epoch: {:02d}'.format(np.mean(valid_losses), valid_auc.result(), best_epoch))
                    f.write('\n [*] Time {:.2f}sec per epoch'.format(time.time() - start))

                    f.close()
                    train_loss_list.append(round(float(np.mean(train_losses)),2))
                    valid_loss_list.append(round(float(np.mean(valid_losses)),2))
                    train_auc_list.append(round(float(train_auc.result()),2))
                    valid_auc_list.append(round(float(valid_auc.result()),2))
                    print(train_loss_list)
                    print(valid_loss_list)
                    print(train_auc_list)
                    print(valid_auc_list)

    def evaluate(self, dataset, filters, n_classes, load_dir_string, data_type_for_prediction_string, hospital_source_type_for_prediction_string):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings(action='ignore')
        self.dataset = dataset
        NUM_GPU = len(utils.get_available_gpus())

        # Model directory
        save_dir_name = FLAGS.load_dir
        checkpoint_prefix = os.path.join(os.path.join('./model', save_dir_name), "ckpt_{epoch}")

        test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        opt = tfa.optimizers.AdamW(weight_decay=FLAGS.wd, learning_rate=FLAGS.lr)

        # Metrics
        test_auc = tf.keras.metrics.AUC()

        def inference_step(batch, model):
            predictions = model([batch[0]['image']], training=False)
            test_auc.update_state(batch[1]['gt60'], predictions['gt60'])
            return predictions['gt60']

        if NUM_GPU == 1:
            model = self.multi_model(filters=filters, n_classes=n_classes)
            model.summary()
            model.load_weights(os.path.join('./model', FLAGS.load_dir, 'best_checkpoint'))
            start = time.time()
            preds_gt60 = []
            for (batch_idx, (batch)) in enumerate(test_data):
                if batch_idx % 50 == 0:
                    print("== {} batch ==".format(batch_idx))
                referral = inference_step(batch, model)
                preds_gt60.extend(referral)

        if not os.path.exists('./preds/{}/'.format(load_dir_string)):
          os.makedirs('./preds/{}/'.format(load_dir_string))
        np.savetxt('./preds/{}/{}_preds_seed{}_{}.out'.format(load_dir_string,data_type_for_prediction_string,FLAGS.random_seed,hospital_source_type_for_prediction_string), np.asarray(preds_gt60))

    def predict(self, dataset, filters, n_classes):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings(action='ignore')
        self.dataset = dataset

        test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        opt = tfa.optimizers.AdamW(weight_decay=FLAGS.wd, learning_rate=FLAGS.lr)
        model = self.multi_model(filters=filters, n_classes=n_classes, training=False)
        model.summary()
        model.load_weights(os.path.join('./model', FLAGS.load_dir, 'best_checkpoint'))

        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy',
                               tf.keras.metrics.AUC(curve='ROC', dtype='float32'),
                               tf.keras.metrics.Recall(dtype='float32'),
                               tf.keras.metrics.Precision(dtype='float32')])

        preds = model.predict(test_data, verbose=1)

        np.save('./preds/feats.npy', preds[0])

    def export(self, dataset, batch, filters, repeat, scales, n_classes):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings(action='ignore')
        self.dataset = dataset
        NUM_GPU = len(utils.get_available_gpus())

        test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        if NUM_GPU == 1:
            model = self.multi_model(filters=filters, n_classes=n_classes, training=False)
            model.compile(optimizer=opt,
                          loss='binary_crossentropy',
                          metrics=['binary_accuracy', tf.keras.metrics.AUC(),
                                   tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

        model.summary()

        model.load_weights(os.path.join('./model', FLAGS.load_dir, 'best_checkpoint'))

        @tf.function
        def full_model(x,y,z):
            tmp = model([x,y,z])
            return [tmp['out60']]


        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
            tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype),
            tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir="./export_model",
                          name="frozen_graph{}.pb".format(FLAGS.random_seed),
                          as_text=False)



    def grad_cam(self, dataset, filters, n_classes, load_dir_string, data_type_for_prediction_string, hospital_source_type_for_prediction_string):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings(action='ignore')
        self.dataset = dataset
        NUM_GPU = len(utils.get_available_gpus())

        # Model directory
        save_dir_name = FLAGS.load_dir

        checkpoint_prefix = os.path.join(os.path.join('./model', save_dir_name), "best_checkpoint")

        # Test data batch generator
        test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Optimizer
        opt = tfa.optimizers.AdamW(weight_decay=FLAGS.wd, learning_rate=FLAGS.lr)#tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
        if NUM_GPU == 1:
            model = self.multi_model(filters=filters, n_classes=n_classes)
            model.summary()
            model.load_weights(checkpoint_prefix.format(epoch=1))

        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = self.multi_model(filters=filters, n_classes=n_classes)

        input_images  = []
        saliency_maps = []

        input_images, saliency_maps = getVanillaSaliencyMap(model, test_data, 'gt60', 'image')

        sample_dir='./saliency/{}/'.format(load_dir_string)
        if not os.path.exists(sample_dir):
          os.makedirs(sample_dir)
        
        np.save(sample_dir+"{}_input_images_{}.npy".format(data_type_for_prediction_string,hospital_source_type_for_prediction_string), np.asarray(input_images))
        np.save(sample_dir+"{}_grad_cam_{}.npy".format(data_type_for_prediction_string,hospital_source_type_for_prediction_string), np.asarray(saliency_maps))


def main(argv):
    del argv
    dataset = DATASETS['fundus']()
    log_width = utils.ilog2(587)
    model = Fundus()
    if FLAGS.mode == 'train':
        model.train(
            dataset=dataset,
            batch=FLAGS.batch,
            filters=FLAGS.filters,
            repeat=FLAGS.repeat,
            scales=FLAGS.scales or (log_width - 5),
            n_classes=1,
            test_case_name=FLAGS.test_case_name
        )
    elif FLAGS.mode == 'eval':
        model.evaluate(
            dataset=dataset,
            filters=FLAGS.filters,
            n_classes=1,
            load_dir_string=FLAGS.load_dir,
            data_type_for_prediction_string=FLAGS.data_type_for_prediction,
            hospital_source_type_for_prediction_string=FLAGS.hospital_source_type_for_prediction
        )
    elif FLAGS.mode == 'predict':
        model.predict(
            dataset=dataset,
            filters=FLAGS.filters,
            n_classes=1
        )
    elif FLAGS.mode == 'grad_cam':
        model.grad_cam(
            dataset=dataset,
            filters=FLAGS.filters,
            n_classes=1,
            load_dir_string=FLAGS.load_dir,
            data_type_for_prediction_string=FLAGS.data_type_for_prediction,
            hospital_source_type_for_prediction_string=FLAGS.hospital_source_type_for_prediction
        )
    else:
        model.export(
            dataset=dataset,
            batch=FLAGS.batch,
            filters=FLAGS.filters,
            repeat=FLAGS.repeat,
            scales=FLAGS.scales or (log_width - 5),
            n_classes=1
        )

if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_string('mode', 'train', '[train or test]')
    flags.DEFINE_string('load_dir', 'y-m-d-h-m-s', 'Load the checkpoint files in this directory')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('epochs', 50, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 160, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    # flags.DEFINE_integer('batch', 32, 'Batch size.')
    flags.DEFINE_integer('batch', 16, 'Batch size.')
    flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
    flags.DEFINE_float('wd', 0.00001, 'Weight decay.')
    flags.DEFINE_string('test_case_name', "not_specified", 'test_case_name for model and pred folders')
    flags.DEFINE_string('data_type_for_prediction', "not_specified", 'train, test, ext')
    flags.DEFINE_string('hospital_source_type_for_prediction', "not_specified", 'CHA, SCHPC')
    app.run(main)
