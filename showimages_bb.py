from datetime import datetime
from dataloader.frames import ActionsDataLoader
from models.unet_acresnet import UNetAc
from models.vision import ResNet50Model
import numpy as np
import tensorflow as tf
import os
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from models.unet_acresnet2skip import UNetAc as UNetAcResNet50_2skips
from models.unet_acresnet import UNetAc as UNetAcResNet50
from models.unet_acresnet0skip import UNetAc as UNetAcResNet50_0skips
flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can AudioCoeff')
flags.DEFINE_string('datatype', 'frames', 'music or outdoor or old')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('batch_size', 2, 'Batch size choose')
flags.DEFINE_integer('plot', 1, 'plot')
flags.DEFINE_integer('nr_frames', 1, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_float('threshold', 0.5, 'threshold')
flags.DEFINE_integer('num_skip_conn', 1, 'Number of skip')
flags.DEFINE_integer('ae', 0, 'auto encoder')
FLAGS = flags.FLAGS

'''compute and plot iou for flickrsoundnet'''

def main(_):

    plotdecodeimages()


def plotdecodeimages():

    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]

    name = '{}_{}_{}_{}'.format(FLAGS.model, dataset, 'AcousticFramesJet2', name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])

    random_pick = True
    build_spectrogram = True
    normalize = False

    # Create data loaders according to the received program arguments
    print('{} - Creating data loaders'.format(datetime.now()))
    modalities = []

    modalities.append(1)
    modalities.append(2)
    plot = FLAGS.plot
    threshold = FLAGS.threshold
    with tf.device('/cpu:0'):
        train_data = ActionsDataLoader(FLAGS.train_file, 'testing', batch_size=FLAGS.batch_size, num_epochs=1, sample_length=1,
                                      datakind=FLAGS.datatype, buffer_size=10, shuffle=False, embedding=1,
                                       normalize=normalize, build_spectrogram=build_spectrogram, correspondence=0,
                                       random_pick=random_pick, modalities=modalities, nr_frames=1)

    # Build model
    print('{} - Building model'.format(datetime.now()))

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
    acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])
    xm = tf.reshape(next_batch[3], shape=[-1, 3])
    xM = tf.reshape(next_batch[4], shape=[-1, 3])
    ym = tf.reshape(next_batch[5], shape=[-1, 3])
    yM = tf.reshape(next_batch[6], shape=[-1, 3])
    typescene = tf.reshape(next_batch[7], shape=[-1, 3])

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
    pos = 0

    print('{} - Starting'.format(datetime.now()))

    namesimage = ['Sound detection', 'Reconstructed']

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
                reconstructed, im, xmin, xmax, ymin, ymax, object = session.run(
                    [output, images, xm, xM, ym, yM, typescene],
                    feed_dict={handle: train_handle,
                               model.network['keep_prob']: 1.0,
                               model.network['is_training']: 0,
                               model_video.network['keep_prob']: 1.0,
                               model_video.network['is_training']: 0
                               })
                total_size += reconstructed.shape[0]

                # if plot:
                #     for h in range(np.shape(reconstructed)[0]):
                #         # original
                #         fig, axs = plt.subplots(2, 2, figsize=(6, 2.9))
                #         plt.tight_layout(pad=1.0)
                #         x = 0
                #         y = 0
                #         imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                #         axs[x, y].imshow(imgray, cmap=plt.cm.gray)
                #         # draw rectangles around contours
                #         m = np.zeros((3, 224, 298), dtype=np.float32)
                #         for contour in range(xmin.shape[1]):
                #             if xmax[h, contour] != 0:
                #                 cv2.rectangle(m[contour], (xmin[h, contour], ymin[h, contour]), (xmax[h, contour], ymax[h, contour]), (255, 255, 255), -1)
                #                 m[contour] = m[contour]/255.
                #                 m[contour] = m[contour]/2.
                #         mtot = np.sum(m, axis=0)
                #         mtot[mtot > 1.0] = 1.0
                #         # m = m * 0.5 + (map > mean) * 0.5
                #
                #         # mbig = cv2.resize(mtot * 1.0, (298, 224))
                #         axs[x, y].imshow(mtot, cmap=plt.cm.viridis, alpha=0.7)
                #         axs[x, y].axis('off')
                #         axs[x, y].set_title('{}'.format(namesimage[0]))
                #         # reconstructed
                #         x = 0
                #         y = 1
                #         imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                #         axs[x, y].imshow(imgray, cmap=plt.cm.gray)
                #         map2 = find_logen(reconstructed[h])
                #         mean2 = np.mean(map2)
                #         std2 = np.std(map2)
                #         m2 = 1 * (map2 > mean2)
                #         # m2 = np.uint8(m2)
                #         # m2 = 1 * (m2 > 0)
                #         # contours, hierarchy = cv2.findContours(m2, cv2.RETR_EXTERNAL,
                #         #                                        cv2.CHAIN_APPROX_SIMPLE)
                #         # big_contour = []
                #         # for con in contours:
                #         #     big_contour.append(con)
                #         # # draw contour
                #         # cv2.drawContours(m2, big_contour, -1, (255, 255, 255), 1)
                #         # # draw rectangles around contours
                #         # for contour in big_contour:
                #         #     (x1, y1, w1, h1) = cv2.boundingRect(contour)
                #         #     cv2.rectangle(m2, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), -1)
                #
                #         m2 = cv2.resize(m2 * 1.0, (298, 224))
                #         m2 = 1.0*(m2>0.5)
                #         axs[x, y].imshow(m2, cmap=plt.cm.viridis, alpha=0.7)
                #         axs[x, y].axis('off')
                #         axs[x, y].set_title('{}'.format(namesimage[1]))
                #
                #         x = 1
                #         y = 1
                #         imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                #         axs[x, y].imshow(imgray, cmap=plt.cm.gray)
                #         intersection = np.logical_and(mtot, m2)
                #         # weight intersection by bounding box weight
                #         intersection = intersection * mtot
                #         axs[x, y].imshow(intersection, cmap=plt.cm.viridis, alpha=0.7)
                #         axs[x, y].axis('off')
                #         axs[x, y].set_title('{}'.format('intersect'))
                #
                #         x = 1
                #         y = 0
                #         imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                #         axs[x, y].imshow(imgray, cmap=plt.cm.gray)
                #         union = np.logical_or(mtot, m2)
                #         # check where is box
                #         box = 1 * (mtot > 0)
                #         # subtract 1-1=0 0.5-1=-0.5
                #         subtract = mtot - box
                #         # area where weight is 0.5 is 1-0.5=0.5  union+(m-1*(m>0))
                #         unionbig = union + subtract
                #         axs[x, y].imshow(unionbig, cmap=plt.cm.viridis, alpha=0.7)
                #         axs[x, y].axis('off')
                #         axs[x, y].set_title('{}'.format('union'))
                #
                #         iou_score = np.sum(intersection) / np.sum(unionbig)
                #         if iou_score > threshold:
                #             pos = pos + 1
                #         outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                #         plt.savefig(outImage_path)
                #         plt.clf()
                #         num = num + 1
                if plot:
                    for h in range(np.shape(reconstructed)[0]):
                        # original
                        # draw rectangles around contours
                        m = np.zeros((3, 224, 298), dtype=np.float32)

                        # mtot = np.sum(m, axis=0)
                        # mtot[mtot > 1.0] = 1.0
                        # reconstructed
                        imgray = cv2.cvtColor(im[h], cv2.COLOR_BGR2GRAY)
                        for contour in range(xmin.shape[1]):
                            if xmax[h, contour] != 0:
                                cv2.rectangle(imgray, (xmin[h, contour], ymin[h, contour]),
                                              (xmax[h, contour], ymax[h, contour]), (1, 1, 1), 3)
                                # m[contour] = m[contour]/255.
                                # m[contour] = m[contour]/2.
                        plt.imshow(imgray, cmap=plt.cm.gray)

                        map2 = find_logen(reconstructed[h])
                        mean2 = np.mean(map2)
                        std2 = np.std(map2)
                        m2 = 1 * (map2 > mean2)

                        m2 = cv2.resize(m2 * 1.0, (298, 224))
                        m2 = 1.0*(m2>0.5)

                        map2 = cv2.resize(map2 * 1.0, (298, 224))
                        plt.imshow(map2, cmap=plt.cm.jet, alpha=0.7)
                        plt.axis('off')

                        # intersection = np.logical_and(mtot, m2)
                        # # weight intersection by bounding box weight
                        # intersection = intersection * mtot
                        #
                        # union = np.logical_or(mtot, m2)
                        # # check where is box
                        # box = 1 * (mtot > 0)
                        # # subtract 1-1=0 0.5-1=-0.5
                        # subtract = mtot - box
                        # # area where weight is 0.5 is 1-0.5=0.5  union+(m-1*(m>0))
                        # unionbig = union + subtract
                        #
                        # iou_score = np.sum(intersection) / np.sum(unionbig)
                        # if iou_score > threshold:
                        #     pos = pos + 1
                        outImage_path = '{}/{}_images_{}.png'.format(data_dir, dataset, num)
                        plt.savefig(outImage_path)
                        plt.clf()
                        num = num + 1
                else:
                    for h in range(np.shape(reconstructed)[0]):
                        m = np.zeros((3, 224, 298), dtype=np.float32)
                        for contour in range(xmin.shape[1]):
                            if xmax[h, contour] != 0:
                                cv2.rectangle(m[contour], (xmin[h, contour], ymin[h, contour]),
                                              (xmax[h, contour], ymax[h, contour]), (255, 255, 255), -1)
                                m[contour] = m[contour] / 255.
                                m[contour] = m[contour] / 2.
                        mtot = np.sum(m, axis=0)
                        mtot[mtot > 1.0] = 1.0

                        # reconstructed
                        map2 = find_logen(reconstructed[h])
                        mean2 = np.mean(map2)
                        std2 = np.std(map2)
                        m2 = 1 * (map2 > mean2)
                        m2 = cv2.resize(m2 * 1.0, (298, 224))
                        m2 = 1.0 * (m2 > 0.5)

                        intersection = np.logical_and(mtot, m2)
                        # weight intersection by bounding box weight
                        intersection = intersection * mtot

                        union = np.logical_or(mtot, m2)
                        # check where is box
                        box = 1 * (mtot > 0)
                        # subtract 1-1=0 0.5-1=-0.5
                        subtract = mtot - box
                        # area where weight is 0.5 is 1-0.5=0.5  union+(m-1*(m>0))
                        unionbig = union + subtract

                        iou_score = np.sum(intersection) / np.sum(unionbig)
                        if iou_score > threshold:
                            pos = pos + 1
                        num = num + 1
                print(total_size)
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
            print(1.0 * pos / num)
        with open('{}'.format(data_dir) + "/intersection_{}_accuracy.txt".format(threshold * 1.0), "w") as outfile:
            outfile.write('iou {:6f}'.format(1.0 * pos / num))

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

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()
