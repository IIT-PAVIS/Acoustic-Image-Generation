from datetime import datetime
from dataloader.outdoor_data_mfcc import ActionsDataLoader as SoundDataLoader
from dataloader.actions_data_old import ActionsDataLoader
from models.unet_acresnet import UNetAc
from models.vision import ResNet50Model
import numpy as np
import tensorflow as tf
import os
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can AudioCoeff')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
FLAGS = flags.FLAGS


def main(_):

    plotdecodeimages()

def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}_{}'.format(FLAGS.model, dataset, 'AcousticMapJet', name)
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
            #variables_in_checkpoint = tf.train.list_variables('path.ckpt')
        session.run(train_iterat.initializer)
        while True:
            try:
                data, reconstructed, im = session.run(
                    [acoustic, output, images],
                    feed_dict={handle: train_handle,
                               model.network['keep_prob']: 1.0,
                               model.network['is_training']: 0,
                               model_video.network['keep_prob']: 1.0,
                               model_video.network['is_training']: 0
                               })
                total_size += reconstructed.shape[0]

                for h in range(np.shape(reconstructed)[0]):
                    # original and reconstructed
                    fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
                    plt.tight_layout(pad=1.0)
                    imagesvideo = np.stack((data, reconstructed), 0)
                    for i in range(2):
                        x = 0
                        y = i
                        imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                        axs[y].imshow(imgray, cmap=plt.cm.gray)
                        map = find_logen(imagesvideo[i, h])
                        map = cv2.resize(map, (298, 224))
                        axs[y].imshow(map, cmap=plt.cm.jet, alpha=0.7)
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

def _build_spectrograms_function(audio_data):
    _NUMBER_OF_SAMPLES = 1024
    n = np.shape(audio_data)[0]
    window = signal.tukey(1024, alpha=0.75)
    window = np.tile(window, (n, 1))
    window = np.reshape(window, (n, _NUMBER_OF_SAMPLES))
    raw_audio = audio_data * window
    fftdata = np.abs(np.fft.rfft(raw_audio, 1024, axis=1))[:, :-1]
    fftdata = fftdata ** 2
    # energy = np.sum(fftdata, axis=-1)
    lifter_num = 22
    lo_freq = 0
    hi_freq = 6400
    filter_num = 24
    mfcc_num = 12
    fft_len = 512

    dct_base = np.zeros((filter_num, mfcc_num))
    for m in range(mfcc_num):
        dct_base[:, m] = np.cos((m + 1) * np.pi / filter_num * (np.arange(filter_num) + 0.5))
    lifter = 1 + (lifter_num / 2) * np.sin(np.pi * (1 + np.arange(mfcc_num)) / lifter_num)

    mfnorm = np.sqrt(2.0 / filter_num)

    filter_mat = createfilters(fft_len, filter_num, lo_freq, hi_freq, 2*hi_freq)
    coefficients = get_feats(fft_len, fftdata, mfcc_num, dct_base, mfnorm, lifter, filter_mat)
    # coefficients[:, 0] = energy
    coefficients = np.float32(coefficients)
    return coefficients

def createfilters(fft_len, filter_num, lo_freq, hi_freq, samp_freq):

    filter_mat = np.zeros((fft_len, filter_num))

    mel2freq = lambda mel: 700.0 * (np.exp(mel / 1127.0) - 1)
    freq2mel = lambda freq: 1127 * (np.log(1 + (freq / 700.0)))

    lo_mel = freq2mel(lo_freq)
    hi_mel = freq2mel(hi_freq)

    mel_c = np.linspace(lo_mel, hi_mel, filter_num + 2)
    freq_c = mel2freq(mel_c)
    # freq_c = np.linspace(lo_freq, hi_freq, filter_num + 2)
    point_c = (freq_c / float(samp_freq) * (fft_len - 1) * 2)
    point_c = np.floor(point_c).astype('int')

    for f in range(filter_num):
        d1 = point_c[f + 1] - point_c[f]
        d2 = point_c[f + 2] - point_c[f + 1]

        filter_mat[point_c[f]:point_c[f + 1] + 1, f] = np.linspace(0, 1, d1 + 1)
        filter_mat[point_c[f + 1]:point_c[f + 2] + 1, f] = np.linspace(1, 0, d2 + 1)

    return filter_mat

def find_logen(mfcc):
    mfcc = np.reshape(mfcc, (-1, 12))

    # lo_freq = 0
    # hi_freq = 6400
    lifter_num = 22
    filter_num = 24
    mfcc_num = 12
    # fft_len = 512
    # filter_mat = createfilters(fft_len, filter_num, lo_freq, hi_freq, 2 * hi_freq)
    dct_base = np.zeros((filter_num, mfcc_num))
    for m in range(mfcc_num):
        dct_base[:, m] = np.cos((m + 1) * np.pi / filter_num * (np.arange(filter_num) + 0.5))
    lifter = 1 + (lifter_num / 2) * np.sin(np.pi * (1 + np.arange(mfcc_num)) / lifter_num)
    mfnorm = np.sqrt(2.0 / filter_num)
    # lifter
    mfcc /= np.expand_dims(lifter, 0)
    mfcc *= mfnorm
    dct_transpose = np.transpose(dct_base)#np.linalg.pinv(dct_base)
    melspec = np.dot(mfcc, dct_transpose)
    # dct_logen = np.cos((1) * np.pi / filter_num * (np.arange(filter_num) + 0.5))
    # logen = np.dot(melspec, dct_logen)
    melspec = np.exp(melspec)

    # filter_mat_pi = np.linalg.pinv(filter_mat)
    # beam = np.dot(melspec, filter_mat_pi)
    sumexpenergies = np.sum(melspec, -1)
    sumexpenergies = 1/sumexpenergies
    map = np.reshape(sumexpenergies, (36, 48))
    return map

def get_feats(fft_len, beam, mfcc_num, dct_base, mfnorm, lifter, filter_mat):
    n = beam.shape[0]
    beam = np.reshape(beam, [n, fft_len])
    # filters
    melspec = np.dot(beam, filter_mat)

    # floor (before log)
    melspec[melspec < 0.001] = 0.001

    # log
    melspec = np.log(melspec)

    # dct
    mfcc_coefficients = np.dot(melspec, dct_base)
    mfcc_coefficients *= mfnorm

    # lifter
    mfcc_coefficients *= lifter

    # sane fixes
    mfcc_coefficients[np.isnan(mfcc_coefficients)] = 0
    mfcc_coefficients[np.isinf(mfcc_coefficients)] = 0

    coefficients = np.reshape(mfcc_coefficients, [n, mfcc_num])

    return coefficients

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()
