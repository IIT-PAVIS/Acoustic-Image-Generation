from datetime import datetime
from dataloader.outdoor_data_mfcc import ActionsDataLoader as SoundDataLoader
from dataloader.actions_data_old import ActionsDataLoader
from models.unet_acresnet import UNetAc as UNetAcResNet50
from models.vision import ResNet50Model
import numpy as np
import tensorflow as tf
import os

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can AudioCoeff')
flags.DEFINE_integer('temporal_pooling', 0, 'Temporal pooling')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_string('encoder_type', 'Video', 'Audio Energy')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('probability', 1, 'Use vae')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
FLAGS = flags.FLAGS

'''Extract features Unet'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}'.format(dataset, FLAGS.encoder_type, name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])

    nr_frames = FLAGS.nr_frames
    random_pick = False
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
            num_actions = 14
            num_locations = 3
            train_data = ActionsDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                      datakind=FLAGS.datatype, buffer_size=10, shuffle=False,
                                       normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                       random_pick=random_pick, modalities=modalities, nr_frames=FLAGS.nr_frames)
        else:
            num_actions = 10
            num_locations = 61
            train_data = SoundDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1,
                                           sample_length=1,
                                           datakind=FLAGS.datatype, buffer_size=10, shuffle=False,
                                           normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                           random_pick=random_pick, modalities=modalities, nr_frames=FLAGS.nr_frames)
    data_size = train_data.num_samples*FLAGS.nr_frames
    # Build model
    print('{} - Building model'.format(datetime.now()))

    with tf.device('/gpu:0'):

        model_video = ResNet50Model(input_shape=[224, 298, 3], num_classes=None)
        model = UNetAcResNet50(input_shape=[36, 48, 12])

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

    labels = tf.reshape(next_batch[3], shape=[-1, num_actions])
    scenario = tf.reshape(next_batch[4], shape=[-1, num_locations])

    model_video._build_model(images)
    model._build_model(mfccmap, model_video.output)

    var_list1 = slim.get_variables(model_video.scope + '/')
    var_list2 = slim.get_variables(model.scope + '/')
    var_list = var_list2 + var_list1

    samples = tf.random_normal([tf.shape(model.std)[0], tf.shape(model.std)[1]], 0, 1, dtype=tf.float32)
    guessed_z = model.mean + (model.std * samples)

    if os.path.exists(data_dir):
        print("Features already computed!")
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories

    dataset_list_features = np.zeros([data_size, 150], dtype=float)
    dataset_labels = np.zeros([data_size, num_actions], dtype=int)
    dataset_scenario = np.zeros([data_size, num_locations], dtype=int)
    print('{} - Starting'.format(datetime.now()))

    total_size = 0
    batch_count = 0
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
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        while True:
            try:
                 labels_data, scenario_data, features = session.run(
                    [labels, scenario, guessed_z],
                    feed_dict={handle: train_handle,
                               model.network['keep_prob']: 1.0,
                               model.network['is_training']: 0,
                               model_video.network['keep_prob']: 1.0,
                               model_video.network['is_training']: 0,
                               })
                 batchnum = labels_data.shape[0]
                 # copy block of data
                 dataset_list_features[total_size:total_size + batchnum, :] = features
                 dataset_labels[total_size:total_size + batchnum, :] = labels_data
                 dataset_scenario[total_size:total_size + batchnum, :] = scenario_data
                 # increase number of data
                 total_size += batchnum
                 end_time = datetime.now()
                 print('{} samples'.format(total_size))
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
    print('{}'.format(data_size))
    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
    np.save('{}/{}_data.npy'.format(data_dir, dataset), dataset_list_features)
    np.save('{}/{}_labels.npy'.format(data_dir, dataset), dataset_labels)
    np.save('{}/{}_scenario.npy'.format(data_dir, dataset), dataset_scenario)

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()