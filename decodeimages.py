from datetime import datetime
from dataloader.outdoor_data import ActionsDataLoader
from models.multimodal import FuseDecoder
from models.unet_sound import UNetSound
from models.unet_architecture_energy import UNetE
from models.unet_z import UNetAc
from models.unet_architecture import UNet
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type Unet')
flags.DEFINE_integer('temporal_pooling', 0, 'Temporal pooling')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('num_classes', 9, 'Number of classes')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('probability', 1, 'Use vae')
flags.DEFINE_string('encoder_type', 'Audio', 'Modality')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_integer('fusion', 0, 'Use both')
FLAGS = flags.FLAGS

'''Plot reconstructed MFCC old fusion'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():
    encoder_type = FLAGS.encoder_type
    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    if FLAGS.fusion:
        name2 = '{}_Ac{}_{}'.format(dataset, 'VideoAudio', name)
    else:
        name2 = '{}_Ac{}_{}'.format(dataset, encoder_type, name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name2])

    random_pick = True

    build_spectrogram = True
    normalize = False

    # Create data loaders according to the received program arguments
    print('{} - Creating data loaders'.format(datetime.now()))
    modalities = []

    modalities.append(0)
    modalities.append(1)
    modalities.append(2)

    with tf.device('/cpu:0'):
        train_data = ActionsDataLoader(FLAGS.train_file, 'inference', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                      datakind='outdoor', buffer_size=10, shuffle=False,
                                       normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                       random_pick=random_pick, modalities=modalities, nr_frames=FLAGS.nr_frames)
    data_size = train_data.num_samples
    # Build model
    print('{} - Building model'.format(datetime.now()))
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    # logenergy = tf.slice(next_batch[0], [0, 0, 0, 0, 0], [-1, 1, 36, 48, 1])
    # logenergy = tf.reshape(logenergy, shape=[-1, 36, 48, 1])
    mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
    images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])

    # logenergy = logenergy - tf.reduce_min(logenergy, axis=[1, 2], keep_dims=True)
    # logenergy = logenergy / tf.reduce_max(logenergy, axis=[1, 2], keep_dims=True)

    # mfcc = mfcc - tf.reduce_min(mfcc, axis=[1, 2], keep_dims=True)
    # mfcc = mfcc / tf.reduce_max(mfcc, axis=[1, 2], keep_dims=True)

    if FLAGS.datatype == 'music':
        num_actions = 9
        num_locations = 11  # maximum number of videos for a class
    else:  # self.datakind == 'outdoor':
        num_actions = 10
        num_locations = 61
    num_embedding = 128
    labels = tf.reshape(next_batch[3], shape=[-1, num_actions])
    scenario = tf.reshape(next_batch[4], shape=[-1, num_locations])

    with tf.device('/gpu:0'):
        modelac = UNetAc(input_shape=[36, 48, 12])
        if FLAGS.fusion:
            modelimages = UNet(input_shape=[224, 298, 3])
            modelimages._build_model(images)
            modelaudio = UNetSound(input_shape=[99, 257, 1])
            modelaudio._build_model(mfcc)
            meanimages = modelimages.mean
            varianceimages = modelimages.variance
            meanaudio = modelaudio.mean
            varianceaudio = modelaudio.variance
            samples = tf.random_normal([tf.shape(varianceimages)[0], tf.shape(varianceimages)[1]], 0, 1, dtype=tf.float32)
            z = meanimages + meanaudio + ((varianceaudio + varianceimages) * samples)
            var_list = slim.get_variables(modelaudio.scope + '/')+slim.get_variables(modelimages.scope + '/')
        else:
            if FLAGS.encoder_type == 'Video':
                model = UNet(input_shape=[224, 298, 3])
                model._build_model(images)
            elif FLAGS.encoder_type == 'Audio':
                model = UNetSound(input_shape=[99, 257, 1])
                model._build_model(mfcc)
            mean = model.mean
            variance = model.variance
            samples = tf.random_normal([tf.shape(variance)[0], tf.shape(variance)[1]], 0, 1, dtype=tf.float32)
            z = mean + (variance * samples)
            var_list = slim.get_variables(model.scope + '/')

        modelac._build_model(acoustic, z)
        output = modelac.output
        var_listac = slim.get_variables(modelac.scope + '/')

    if os.path.exists(data_dir):
        print("Features already computed!")
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories

    total_size = 0
    batch_count = 0
    num = 0
    print('{} - Starting'.format(datetime.now()))


    namesimage = ['Acoustic image', 'Reconstructed']

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        saver = tf.train.Saver(var_list=var_listac + var_list)
        saver.restore(session, FLAGS.init_checkpoint)
        print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        if FLAGS.fusion:
            while True:
                try:
                    data, reconstructed = session.run(
                        [acoustic, output],
                        feed_dict={handle: train_handle,
                                   modelac.network['keep_prob']: 1.0,
                                   modelac.network['is_training']: 0,
                                   modelaudio.network['keep_prob']: 1.0,
                                   modelaudio.network['is_training']: 0,
                                   modelimages.network['keep_prob']: 1.0,
                                   modelimages.network['is_training']: 0})
                    total_size += reconstructed.shape[0]

                    for h in range(np.shape(reconstructed)[0]):
                        # original and reconstructed
                        fig, axs = plt.subplots(4, 2, figsize=(6, 2.9 * 4))
                        plt.tight_layout(pad=1.0)
                        fig.suptitle('Reconstructed image')
                        imagesvideo = np.stack((data, reconstructed), 0)
                        for i in range(2):
                            for j in range(4):
                                x = j
                                y = i
                                axs[x, y].imshow(imagesvideo[i, h, :, :, j * 3:(j + 1) * 3])
                                axs[x, y].axis('off')
                                axs[x, y].set_title('{}'.format(namesimage[i]))
                        outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                        plt.savefig(outImage_path)
                        plt.clf()
                        num = num + 1
                    print('{} samples'.format(total_size))
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
        else:
            while True:
                try:
                     data, reconstructed = session.run(
                        [acoustic, output],
                        feed_dict={handle: train_handle,
                                   modelac.network['keep_prob']: 1.0,
                                   modelac.network['is_training']: 0,
                                   model.network['keep_prob']: 1.0,
                                   model.network['is_training']: 0})
                     total_size += reconstructed.shape[0]

                     for h in range(np.shape(reconstructed)[0]):
                         # original and reconstructed
                         fig, axs = plt.subplots(4, 2, figsize=(6, 2.9 * 4))
                         plt.tight_layout(pad=1.0)
                         fig.suptitle('Reconstructed image')
                         imagesvideo = np.stack((data, reconstructed), 0)
                         for i in range(2):
                             for j in range(4):
                                 x = j
                                 y = i
                                 axs[x, y].imshow(imagesvideo[i, h, :, :, j * 3:(j + 1) * 3])
                                 axs[x, y].axis('off')
                                 axs[x, y].set_title('{}'.format(namesimage[i]))
                         outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                         plt.savefig(outImage_path)
                         plt.clf()
                         num = num + 1
                     print('{} samples'.format(total_size))
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1

    print('{}'.format(data_size))
    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()