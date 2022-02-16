from datetime import datetime
from dataloader.outdoor_data_mfcc import ActionsDataLoader as SoundDataLoader
from dataloader.actions_data_old import ActionsDataLoader
from models.unet_acresnet import UNetAc
from models.vision import ResNet50Model
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can AudioCoeff')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
FLAGS = flags.FLAGS

'''Plot reconstructed MFCC'''

def main(_):

    plotdecodeimages()

def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}_{}'.format(FLAGS.model, dataset, 'Acoustic', name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])
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
        if FLAGS.datatype == 'old':
            train_data = ActionsDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                          datakind=FLAGS.datatype, buffer_size=10, shuffle=False, embedding=1,
                                           normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                           random_pick=random_pick, modalities=modalities, nr_frames=1)
        elif FLAGS.datatype == 'outdoor':
            train_data = SoundDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                          datakind=FLAGS.datatype, buffer_size=10, shuffle=False, embedding=1,
                                           normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                           random_pick=random_pick, modalities=modalities, nr_frames=1)

    # Build model
    print('{} - Building model'.format(datetime.now()))

    with tf.device('/gpu:0'):

        model = UNetAc(input_shape=[36, 48, 12])
        model_video = ResNet50Model(input_shape=[224, 298, 3], num_classes=None)

    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    mfcc = tf.reshape(next_batch[1], shape=[-1, 12])
    images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])

    # mfcc = mfcc - tf.reduce_min(mfcc, axis=[1], keep_dims=True)
    # mfcc = mfcc / tf.reduce_max(mfcc, axis=[1], keep_dims=True)

    mfccmap = tf.reshape(mfcc, (-1, 1, 12))
    mfccmap = tf.tile(mfccmap, (1, 36 * 48, 1))
    mfccmap = tf.reshape(mfccmap, (-1, 36, 48, 12))

    model_video._build_model(images)
    model._build_model(mfccmap, model_video.output)

    output = model.output
    var_list1 = slim.get_variables(model_video.scope + '/')
    var_list2 = slim.get_variables(model.scope + '/')
    var_list = var_list2 + var_list1

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
        # Initialize student model
        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            model.init_model(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables(FLAGS.init_checkpoint)
        session.run(train_iterat.initializer)
        while True:
            try:
                data, reconstructed = session.run(
                    [acoustic, output],
                    feed_dict={handle: train_handle,
                               model.network['keep_prob']: 1.0,
                               model.network['is_training']: 0,
                               model_video.network['keep_prob']: 1.0,
                               model_video.network['is_training']: 0
                               })
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
                print(total_size)
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
            print('{} - Completed, got {} samples'.format(datetime.now(), total_size))

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()
