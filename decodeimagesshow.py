from datetime import datetime
from dataloader.outdoor_data import ActionsDataLoader
from models.unet_sound2 import UNetSound
from models.unet_architecture_energy import UNetE
from models.unet_noconc import UNetAc
from models.unet_architecture_noconc import UNet
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can AudioCoeff')
flags.DEFINE_integer('temporal_pooling', 0, 'Temporal pooling')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_string('encoder_type', 'Video', 'Audio Energy')
flags.DEFINE_integer('num_classes', 9, 'Number of classes')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('probability', 1, 'Use vae')
FLAGS = flags.FLAGS

'''Plot energy old'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}_{}'.format(FLAGS.model, dataset, FLAGS.encoder_type, name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])
    num_classes = FLAGS.num_classes
    temporal_pooling = FLAGS.temporal_pooling

    nr_frames = FLAGS.nr_frames
    random_pick = True

    build_spectrogram = (FLAGS.model == 'AudioCoefficients' or FLAGS.model == 'ResNet50' or FLAGS.model == 'HearNet'
                         or FLAGS.model == 'UNet' or FLAGS.model == 'ResNet18_v1')
    normalize = FLAGS.model == 'HearNet'

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

    # Build model
    print('{} - Building model'.format(datetime.now()))

    with tf.device('/gpu:0'):

        if FLAGS.encoder_type == 'Video':
            model = UNet(input_shape=[224, 298, 3])
        elif FLAGS.encoder_type == 'Audio':
            model = UNetSound(input_shape=[99, 257, 1])
        elif FLAGS.encoder_type == 'Ac':
            model = UNetAc(input_shape=[36, 48, 12])
        else:
            model = UNetE(input_shape=[36, 48, 1])
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    logenergy = tf.slice(next_batch[0], [0, 0, 0, 0, 0], [-1, 1, 36, 48, 1])
    logenergy = tf.reshape(logenergy, shape=[-1, 36, 48, 1])
    mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
    mfcc = tf.image.resize_bilinear(mfcc, [193, 257], align_corners=False)
    images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])

    logenergy = logenergy - tf.reduce_min(logenergy, axis=[1, 2], keep_dims=True)
    logenergy = logenergy / tf.reduce_max(logenergy, axis=[1, 2], keep_dims=True)

    # mfcc = mfcc - tf.reduce_min(mfcc, axis=[1, 2], keep_dims=True)
    # mfcc = mfcc / tf.reduce_max(mfcc, axis=[1, 2], keep_dims=True)

    if FLAGS.encoder_type == 'Video':
        considered_modality = images
    elif FLAGS.encoder_type == 'Audio':
        considered_modality = mfcc
    elif FLAGS.encoder_type == 'Ac':
        considered_modality = acoustic
    else:
        considered_modality = logenergy

    model._build_model(considered_modality)
    #FLAGS.model == 'UNet'
    output = model.output
    var_list2 = slim.get_variables(model.scope + '/')

    if os.path.exists(data_dir):
        print("Features already computed!")
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories

    total_size = 0
    batch_count = 0
    num = 0
    print('{} - Starting'.format(datetime.now()))

    if FLAGS.encoder_type == 'Video':
        namesimage = ['RGB', 'Reconstructed']
    elif FLAGS.encoder_type == 'Audio':
        namesimage = ['Spectrogram', 'Reconstructed']
    elif FLAGS.encoder_type == 'Ac':
        namesimage = ['Acoustic image', 'Reconstructed']
    else:
        namesimage = ['Energy', 'Reconstructed']

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        # Initialize student model
        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            model.init_model(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))
            saver = tf.train.Saver(var_list=var_list2)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        if FLAGS.encoder_type == 'Audio' or FLAGS.encoder_type == 'Energy':
            while True:
                try:
                     data, reconstructed = session.run(
                        [considered_modality, output],
                        feed_dict={handle: train_handle,
                                   model.network['keep_prob']: 1.0,
                                   model.network['is_training']: 0})
                     total_size += reconstructed.shape[0]

                     for h in range(np.shape(reconstructed)[0]):
                         # original and reconstructed
                         fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
                         plt.tight_layout(pad=1.0)
                         fig.suptitle('Reconstructed image')
                         imagesvideo = np.stack((data, reconstructed), 0)
                         for i in range(2):
                             x = 0
                             y = i
                             axs[y].imshow(imagesvideo[i, h, :, :, 0])
                             axs[y].axis('off')
                             axs[y].set_title('{}'.format(namesimage[i]))
                         outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                         plt.savefig(outImage_path)
                         plt.clf()
                         num = num + 1
                     print(total_size)
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
                print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
        elif FLAGS.encoder_type == 'Video':
            while True:
                try:
                     data, reconstructed = session.run(
                        [considered_modality, output],
                        feed_dict={handle: train_handle,
                                   model.network['keep_prob']: 1.0,
                                   model.network['is_training']: 0})
                     total_size += reconstructed.shape[0]

                     for h in range(np.shape(reconstructed)[0]):
                         # original and reconstructed
                         fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
                         plt.tight_layout(pad=1.0)
                         fig.suptitle('Reconstructed image')
                         imagesvideo = np.stack((data, reconstructed), 0)
                         for i in range(2):
                             x = 0
                             y = i
                             axs[y].imshow(imagesvideo[i, h, :, :, :])
                             axs[y].axis('off')
                             axs[y].set_title('{}'.format(namesimage[i]))
                         outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                         plt.savefig(outImage_path)
                         plt.clf()
                         num = num + 1
                     print(total_size)
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
                print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
        else:
            while True:
                try:
                    data, reconstructed = session.run(
                        [considered_modality, output],
                        feed_dict={handle: train_handle,
                                   model.network['keep_prob']: 1.0,
                                   model.network['is_training']: 0})
                    total_size += reconstructed.shape[0]

                    for h in range(np.shape(reconstructed)[0]):
                        # original and reconstructed
                        fig, axs = plt.subplots(4, 2, figsize=(6, 2.9*4))
                        plt.tight_layout(pad=1.0)
                        fig.suptitle('Reconstructed image')
                        imagesvideo = np.stack((data, reconstructed), 0)
                        for i in range(2):
                            for j in range(4):
                                x = j
                                y = i
                                axs[x, y].imshow(imagesvideo[i, h, :, :, j*3:(j+1)*3])
                                axs[x, y].axis('off')
                                axs[x, y].set_title('{}'.format(namesimage[i]))
                        outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                        plt.savefig(outImage_path)
                        plt.clf()
                        num = num + 1
                    print(total_size)
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
                print('{} - Completed, got {} samples'.format(datetime.now(), total_size))

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()