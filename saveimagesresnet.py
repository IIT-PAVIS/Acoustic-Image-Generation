from datetime import datetime
from dataloader.outdoor_data_mfcc import ActionsDataLoader as SoundDataLoader
from dataloader.actions_data_old import ActionsDataLoader
from models.dualcamnet import DualCamHybridModel
from models.unet_acresnet2skip import UNetAc as UNetAcResNet50_2skips
from models.unet_acresnet import UNetAc as UNetAcResNet50
from models.unet_acresnet0skip import UNetAc as UNetAcResNet50_0skips
from models.vision import ResNet50Model
import numpy as np
import tensorflow as tf
import os
from models.base import buildAccuracy

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for unet initialization')
flags.DEFINE_string('ac_checkpoint', None, 'Checkpoint file for dual cam net initialization')
flags.DEFINE_integer('batch_size', 16, 'Batch size choose')
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('mfccmap', 0, 'Do not reconstruct')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_integer('num_skip_conn', 1, 'Number of skip')
flags.DEFINE_integer('ae', 0, 'auto encoder')
FLAGS = flags.FLAGS

'''Compute accuracy on acoustic images and generated'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    s = FLAGS.ac_checkpoint.split('/')[-1]
    nameac = (s.split('_')[1]).split('.ckpt')[0]

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
            num_classes = 14
            train_data = ActionsDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                          datakind=FLAGS.datatype, buffer_size=10, shuffle=False, embedding=0,
                                           normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                           random_pick=random_pick, modalities=modalities, nr_frames=12)
        elif FLAGS.datatype=='outdoor':
            num_classes = 10
            train_data = SoundDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                          datakind=FLAGS.datatype, buffer_size=10, shuffle=False, embedding=0,
                                           normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                           random_pick=random_pick, modalities=modalities, nr_frames=12)

    modelacustic = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=num_classes, embedding=0)
    modelnegative = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=num_classes, embedding=0)
    data_size = train_data.num_samples
    # Build model
    print('{} - Building model'.format(datetime.now()))
    print(data_size)
    with tf.device('/gpu:0'):
        model_video = ResNet50Model(input_shape=[224, 298, 3], num_classes=None)
        if FLAGS.num_skip_conn == 2:
            model = UNetAcResNet50_2skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)
        elif FLAGS.num_skip_conn == 1:
            model = UNetAcResNet50(input_shape=[36, 48, 12], embedding=FLAGS.ae)
        elif FLAGS.num_skip_conn == 0:
            model = UNetAcResNet50_0skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)


    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    mfcc = tf.reshape(next_batch[1], shape=[-1, 12])
    images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 12, 36, 48, 12])

    # mfcc = mfcc - tf.reduce_min(mfcc, axis=[1], keep_dims=True)
    # mfcc = mfcc / tf.reduce_max(mfcc, axis=[1], keep_dims=True)

    mfccmap = tf.reshape(mfcc, (-1, 12, 1, 12))
    mfccmap = tf.tile(mfccmap, (1, 1, 36 * 48, 1))
    mfccmap = tf.reshape(mfccmap, (-1, 36, 48, 12))
    model_video._build_model(images)
    model._build_model(mfccmap, model_video.output)

    modelacustic._build_model(acoustic)
    labels = tf.reshape(next_batch[3], shape=[-1, num_classes])
    if FLAGS.mfccmap == 0:
        output = tf.reshape(model.output, shape=[-1, 12, 36, 48, 12])
    else:
        output = tf.reshape(mfccmap, shape=[-1, 12, 36, 48, 12])

    # if os.path.exists(data_dir):
    #     print("Features already computed!")
    # else:
    #     os.makedirs(data_dir)
    modelnegative._build_model(output)

    expanded_shape = [-1, 12, num_classes]
    logitsacoustic = tf.reduce_mean(tf.reshape(modelacustic.output, shape=expanded_shape), axis=1)
    logistnegative = tf.reduce_mean(tf.reshape(modelnegative.output, shape=expanded_shape), axis=1)
    accuracyacoustic = buildAccuracy(logitsacoustic, labels)
    accuracynegative = buildAccuracy(logistnegative, labels)

    total_size = 0
    batch_count = 0
    num = 0
    accuracyac = 0
    accuracyfalse = 0
    # dataset_list_images = np.zeros([data_size, 36, 48, 12], dtype=float)
    # dataset_list_acoustic = np.zeros([data_size, 36, 48, 12], dtype=float)
    print('{} - Starting'.format(datetime.now()))
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        # Initialize student model
        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # latest_ckp = FLAGS.init_checkpoint
        # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='resnet_v1_50/conv_map/BatchNorm/gamma')
        # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='resnet_v1_50/conv_map/BatchNorm/moving_variance')
        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            model.init_model(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))

            var_list1 = slim.get_variables(model_video.scope + '/')
            var_list2 = slim.get_variables(model.scope + '/')
            var_list = var_list2 + var_list1

            # to_exclude = [i.name for i in tf.global_variables()
            #               if modelacustic.scope in i.name or 'moving_mean' in i.name or 'moving_variance' in i.name or
            #               '/Adam' in i.name or 'power' in i.name or 'step' in i.name]
            # # or 'vgg_vox' in i.name
            # var_list = slim.get_variables_to_restore(exclude=to_exclude)
            # Attempting
            # to
            # use
            # uninitialized
            # value
            # resnet_v1_50 / block3 / unit_6 / bottleneck_v1 / conv2 / BatchNorm / moving_mean

            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
            var_list = slim.get_variables(modelacustic.scope + '/')
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.ac_checkpoint)
            var_list = slim.get_variables(modelnegative.scope + '/')
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.ac_checkpoint)
        session.run(train_iterat.initializer)
        while True:
            try:
                # reconstructed, ac = session.run([output, acoustic],
                #     feed_dict={handle: train_handle,
                #                model.network['keep_prob']: 1.0,
                #                model.network['is_training']: 0,
                #                model_video.network['keep_prob']: 1.0,
                #                model_video.network['is_training']: 0})
                # batchnum = reconstructed.shape[0]
                # dataset_list_images[total_size:total_size + batchnum, :] = reconstructed
                # dataset_list_acoustic[total_size:total_size + batchnum, :] = ac

                # ac = np.expand_dims(ac, axis=1)
                # ac = np.tile(ac, (1, 12, 1, 1, 1))
                # reconstructed = np.expand_dims(reconstructed, axis=1)
                # reconstructed = np.tile(reconstructed, (1, 12, 1, 1, 1))
                acc, accrec, labelsvalue = session.run([accuracyacoustic, accuracynegative, labels], feed_dict={
                    handle: train_handle,
                    model.network['keep_prob']: 1.0,
                    model.network['is_training']: 0,
                    model_video.network['keep_prob']: 1.0,
                    model_video.network['is_training']: 0,
                             modelnegative.network['keep_prob']: 1,
                             modelnegative.network['is_training']: 0,
                             modelacustic.network['keep_prob']: 1,
                             modelacustic.network['is_training']: 0})

                total_size += labelsvalue.shape[0]
                accuracyac += acc * labelsvalue.shape[0]
                accuracyfalse += accrec * labelsvalue.shape[0]
                print(total_size)
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
        # np.save('{}/ac.npy'.format(data_dir), dataset_list_acoustic)
        # np.save('{}/acreconstructed.npy'.format(data_dir), dataset_list_images)
        print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
        acctot = accuracyac/total_size
        accrectot = accuracyfalse/total_size
        print('acc rec {} acc ac {}'.format(accrectot, acctot))
        if FLAGS.mfccmap == 0:
            with open('{}'.format(str.join('/', FLAGS.init_checkpoint.split('/')[:-1])) + "/test_unet{}_dualcamnet{}.txt".format(name, nameac), "w") as outfile:
                outfile.write('acc rec {} acc ac {}'.format(accrectot, acctot))
        else:
            with open('{}'.format(str.join('/', FLAGS.ac_checkpoint.split('/')[:-1])) + "/test_map_dualcamnet{}.txt".format(nameac), "w") as outfile:
                outfile.write('acc rec {} acc ac {}'.format(accrectot, acctot))


if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()