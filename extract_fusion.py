from datetime import datetime
from dataloader.outdoor_data import ActionsDataLoader
from models.multimodal import AssociatorVideoAc
from models.multimodal import AssociatorAudioAc
from models.multimodal import AssociatorAudio
from models.unet_sound2 import UNetSound
from models.unet_z import UNetAc
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
flags.DEFINE_integer('num_classes', 9, 'Number of classes')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('onlyaudio', 0, 'Using only audio associator no sound net')
flags.DEFINE_integer('probability', 1, 'Use vae')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
FLAGS = flags.FLAGS

'''Extract features old'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    nameac = '{}_{}_{}'.format(dataset, 'Ac', name)
    nameaudio = '{}_{}_{}'.format(dataset, 'Audio', name)
    # nameimages = '{}_{}_{}'.format(dataset, 'Video', name)
    # nameimagesaudio = '{}_{}_{}'.format(dataset, 'VideoAudio', name)
    data_dirac = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [nameac])
    data_diraudio = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [nameaudio])
    # data_dirimages = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [nameimages])
    # data_dirimagesaudio = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [nameimagesaudio])
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
    data_size = train_data.num_samples
    # Build model
    print('{} - Building model'.format(datetime.now()))

    with tf.device('/gpu:0'):

        modelimages = UNet(input_shape=[224, 298, 3])
        modelaudio = UNetSound(input_shape=[99, 257, 1])
        modelac = UNetAc(input_shape=[36, 48, 12])
        # model_associator = AssociatorVideoAc(input_shape=1024)
        if FLAGS.onlyaudio:
            model_associator1 = AssociatorAudio(input_shape=[193, 257, 1])
        else:
            model_associator1 = AssociatorAudioAc(input_shape=256)
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
    mfcc = tf.image.resize_bilinear(mfcc, [193, 257], align_corners=False)
    # images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])

    # mfcc = mfcc - tf.reduce_min(mfcc, axis=[1, 2], keep_dims=True)
    # mfcc = mfcc / tf.reduce_max(mfcc, axis=[1, 2], keep_dims=True)

    if FLAGS.datatype == 'music':
        num_actions = 9
        num_locations = 11  # maximum number of videos for a class
    else:  # self.datakind == 'outdoor':
        num_actions = 10
        num_locations = 61
    num_embedding = 150
    labels = tf.reshape(next_batch[3], shape=[-1, num_actions])
    scenario = tf.reshape(next_batch[4], shape=[-1, num_locations])
    # modelimages._build_model(images)
    # model_associator._build_model(modelimages.mean, modelimages.std)
    if not FLAGS.onlyaudio:
        modelaudio._build_model(mfcc)
        model_associator1._build_model(modelaudio.mean, modelaudio.std)
    else:
        model_associator1._build_model(mfcc)
    mean = model_associator1.mean
    std = model_associator1.std
    modelac._build_model(acoustic, mean, std)


    samples = tf.random_normal([tf.shape(modelac.mean)[0], tf.shape(modelac.mean)[1]], 0, 1,
                               dtype=tf.float32)
    # guessed_z = model.mean + (model.variance * samples)
    extractedac = modelac.mean + (modelac.std * samples)
    extractedaudio = model_associator1.mean + (model_associator1.std * samples)
    # extractedvideo = model_associator.mean + (model_associator.std * samples)
    extractedvideoaudio = mean + (std * samples)
    #FLAGS.model == 'UNet'
    var_listac = slim.get_variables(modelac.scope + '/')
    var_listaudio = slim.get_variables(modelaudio.scope + '/')
    # var_listimages = slim.get_variables(modelimages.scope + '/')
    # var_listassociatorimages = slim.get_variables(model_associator.scope + '/')
    var_listassociatoraudio = slim.get_variables(model_associator1.scope + '/')

    if os.path.exists(data_dirac):
        print("Features already computed!")
    else:
        os.makedirs(data_dirac)

    if os.path.exists(data_diraudio):
        print("Features already computed!")
    else:
        os.makedirs(data_diraudio)

    # if os.path.exists(data_dirimages):
    #     print("Features already computed!")
    # else:
    #     os.makedirs(data_dirimages)
    #
    # if os.path.exists(data_dirimagesaudio):
    #     print("Features already computed!")
    # else:
    #     os.makedirs(data_dirimagesaudio)

    num = 0
    total_size = 0
    batch_count = 0

    dataset_list_featuresac = np.zeros([data_size, num_embedding], dtype=float)
    dataset_list_featuresaudio = np.zeros([data_size, num_embedding], dtype=float)
    # dataset_list_featuresimages = np.zeros([data_size, num_embedding], dtype=float)
    # dataset_list_featuresimagesaudio = np.zeros([data_size, num_embedding], dtype=float)
    dataset_labels = np.zeros([data_size, num_actions], dtype=int)
    dataset_scenario = np.zeros([data_size, num_locations], dtype=int)

    print('{} - Starting'.format(datetime.now()))

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        saver = tf.train.Saver(var_list=var_listac + var_listaudio + var_listassociatoraudio)
        saver.restore(session, FLAGS.init_checkpoint)
        print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        if FLAGS.onlyaudio:
            while True:
                try:
                    labels_data, scenario_data, featuresac, featuresaudio = session.run(
                        [labels, scenario, extractedac, extractedaudio],
                        feed_dict={handle: train_handle,
                                   modelac.network['keep_prob']: 1.0,
                                   modelac.network['is_training']: 0,
                                   model_associator1.network['keep_prob']: 1.0,
                                   model_associator1.network['is_training']: 0})
                    batchnum = labels_data.shape[0]
                    # copy block of data
                    # dataset_list_featuresimages[total_size:total_size + batchnum, :] = featuresimages
                    # dataset_list_featuresimagesaudio[total_size:total_size + batchnum, :] = featuresimagesaudio
                    dataset_list_featuresaudio[total_size:total_size + batchnum, :] = featuresaudio
                    dataset_list_featuresac[total_size:total_size + batchnum, :] = featuresac
                    dataset_labels[total_size:total_size + batchnum, :] = labels_data
                    dataset_scenario[total_size:total_size + batchnum, :] = scenario_data
                    # increase number of data
                    total_size += batchnum
                    print('{} samples'.format(total_size))
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
        else:
            while True:
                try:
                     labels_data, scenario_data, featuresac, featuresaudio = session.run(
                        [labels, scenario, extractedac, extractedaudio],
                        feed_dict={handle: train_handle,
                                   modelac.network['keep_prob']: 1.0,
                                   modelac.network['is_training']: 0,
                                   modelaudio.network['keep_prob']: 1.0,
                                   modelaudio.network['is_training']: 0})
                     batchnum = labels_data.shape[0]
                     # copy block of data
                     # dataset_list_featuresimages[total_size:total_size + batchnum, :] = featuresimages
                     # dataset_list_featuresimagesaudio[total_size:total_size + batchnum, :] = featuresimagesaudio
                     dataset_list_featuresaudio[total_size:total_size + batchnum, :] = featuresaudio
                     dataset_list_featuresac[total_size:total_size + batchnum, :] = featuresac
                     dataset_labels[total_size:total_size + batchnum, :] = labels_data
                     dataset_scenario[total_size:total_size + batchnum, :] = scenario_data
                     # increase number of data
                     total_size += batchnum
                     print('{} samples'.format(total_size))
                except tf.errors.OutOfRangeError:
                    break
                batch_count += 1
    print('{}'.format(data_size))
    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
    np.save('{}/{}_data.npy'.format(data_dirac, dataset), dataset_list_featuresac)
    np.save('{}/{}_labels.npy'.format(data_dirac, dataset), dataset_labels)
    np.save('{}/{}_scenario.npy'.format(data_dirac, dataset), dataset_scenario)

    np.save('{}/{}_data.npy'.format(data_diraudio, dataset), dataset_list_featuresaudio)
    np.save('{}/{}_labels.npy'.format(data_diraudio, dataset), dataset_labels)
    np.save('{}/{}_scenario.npy'.format(data_diraudio, dataset), dataset_scenario)

    # np.save('{}/{}_data.npy'.format(data_dirimages, dataset), dataset_list_featuresimages)
    # np.save('{}/{}_labels.npy'.format(data_dirimages, dataset), dataset_labels)
    # np.save('{}/{}_scenario.npy'.format(data_dirimages, dataset), dataset_scenario)
    #
    # np.save('{}/{}_data.npy'.format(data_dirimagesaudio, dataset), dataset_list_featuresimagesaudio)
    # np.save('{}/{}_labels.npy'.format(data_dirimagesaudio, dataset), dataset_labels)
    # np.save('{}/{}_scenario.npy'.format(data_dirimagesaudio, dataset), dataset_scenario)

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()