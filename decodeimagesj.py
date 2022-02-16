from datetime import datetime
from dataloader.outdoor_data import ActionsDataLoader
from models.multimodal import Jointmvae
from models.multimodal import JointTwomvae2
from models.multimodal import JointTwomvae
from models.unet_sound22 import UNetSound
from models.unet_noconc2 import UNetAc
from models.unet_architecture_noconc2 import UNet
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
flags.DEFINE_integer('probability', 1, 'Use vae')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_integer('fusion', 0, 'Use both audio and video')
flags.DEFINE_integer('onlyaudiovideo', 0, 'Using only audio and video')
FLAGS = flags.FLAGS

'''Plot energy old'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}_{}'.format(FLAGS.model, dataset, 'Ac', name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])
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
        if FLAGS.fusion:
            model_associator = JointTwomvae2()
        elif FLAGS.onlyaudiovideo:
            model_associator = JointTwomvae()
        else:
            model_associator = Jointmvae()
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
    mfcc = tf.image.resize_bilinear(mfcc, [193, 257], align_corners=False)
    video = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])

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
    output = modelac._build_network(acoustic)
    outputvideo = modelimages._build_network(video)
    outputaudio = modelaudio._build_network(mfcc)
    # fuse feature maps and get new feature maps for 3 mod
    if FLAGS.fusion or FLAGS.onlyaudiovideo:
        model_associator._build_model(outputvideo, outputaudio)
    else:
        model_associator._build_model(output, outputvideo, outputaudio)

    if FLAGS.onlyaudiovideo:
        modelac._build_model(model_associator.outputac)
    else:
        modelac._build_model(model_associator.outputac)
        modelaudio._build_model(model_associator.outputaudio)
        modelimages._build_model(model_associator.outputvideo)

    #FLAGS.model == 'UNet'
    var_listac = slim.get_variables(modelac.scope + '/')
    var_listaudio = slim.get_variables(modelaudio.scope + '/')
    var_listimages = slim.get_variables(modelimages.scope + '/')
    var_listassociator = slim.get_variables(model_associator.scope + '/')

    if os.path.exists(data_dir):
        print("Features already computed!")
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories
    num = 0
    total_size = 0
    batch_count = 0

    print('{} - Starting'.format(datetime.now()))

    namesimage = ['Acoustic image', 'Reconstructed']

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        saver = tf.train.Saver(var_list=var_listac + var_listaudio + var_listimages + var_listassociator)
        saver.restore(session, FLAGS.init_checkpoint)
        print('{} - Done'.format(datetime.now()))
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        while True:
            try:
                 data, reconstructed = session.run(
                    [acoustic, modelac.output],
                    feed_dict={handle: train_handle,
                               modelac.keep_prob: 1.0,
                               modelac.is_training: 0,
                               modelaudio.keep_prob: 1.0,
                               modelaudio.is_training: 0,
                               modelimages.keep_prob: 1.0,
                               modelimages.is_training: 0})
                 batchnum = reconstructed.shape[0]
                 # copy block of data
                 # increase number of data
                 total_size += batchnum
                 print('{} samples'.format(total_size))
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
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
    print('{}'.format(data_size))
    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()
