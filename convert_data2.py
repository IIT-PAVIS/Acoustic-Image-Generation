import argparse
import cv2
import glob
import math
import numpy as np
import os
import re
import librosa
import tensorflow as tf
from collections import namedtuple
from datetime import datetime
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import xml.etree.ElementTree as ET

Image = namedtuple('Image', 'rows cols depth data')
Audio = namedtuple('Audio', 'mics samples data')

_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12

'''Create tfrecords for FlickrSoundNet'''

def one_microphone(audio_data):
    """Python function to build a waveform audio from audio samples."""
    # choose index
    mic_id = 0
    # consider audio of one microphone
    audio_data_mic = audio_data[mic_id, :]
    return audio_data_mic
def _read_wav_audio_data(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))
    fs, data = wavfile.read(filename)
    data2 = librosa.core.resample(data*1.0, fs, 12*1024)
    # c = np.abs(np.fft.rfft(data2, axis=0))[:-1]
    # d = len(c)#half of the fft list (real signal symmetry)
    # frequencies = np.linspace(0, 6 * 1024, d)
    # plt.plot(frequencies, abs(c), 'r')
    # plt.show()
    # seconds = len(data2)//(1024)
    # for i in range(seconds):
    #     c = np.abs(np.fft.rfft(data2[i*1024:(i+1)*1024], axis=0))[:-1]
    #     d = len(c)#half of the fft list (real signal symmetry)
    #     frequencies = np.linspace(0, 6 * 1024, d)
    #     plt.plot(frequencies, abs(c), 'r')
    #     plt.show()
    data2 = np.int32(data2)
    return data2

def _read_raw_audio_data(audio_data_sample):
    audio_serialized = audio_data_sample.tostring()

    return Audio(mics=1, samples=len(audio_data_sample), data=audio_serialized)


def str2dir(dir_name):
    if not os.path.isdir(dir_name):
        raise argparse.ArgumentTypeError('{} is not a directory!'.format(dir_name))
    # elif os.access(dirname, os.R_OK):
    #     return argparse.ArgumentTypeError('{} is not a readable directory!'.format(dirname))
    else:
        return os.path.abspath(os.path.expanduser(dir_name))


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image.
      smallest_side: A python integer or scalar indicating the size of
        the smallest side after resize.
    Returns:
      resized_image: A 3-D resized image.
    """
    shape = np.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar indicating the current height.
      width: an int32 scalar indicating the current width.
      smallest_side: A python integer or scalar indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar indicating the new height.
      new_width: and int32 scalar indicating the new width.
    """
    height = float(height)
    width = float(width)
    smallest_side = float(smallest_side)

    if height > width:
        scale = smallest_side / width
    else:
        scale = smallest_side / height
    new_height = int(height * scale)
    new_width = int(width * scale)
    return new_height, new_width


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar indicating the height offset.
      offset_width: a scalar indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = np.shape(image)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = image[offset_height:crop_height + offset_height, offset_width:crop_width + offset_width, :]
    return image


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image.
    Args:
      image: image.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      cropped image.
    """
    outputs = []
    image_height = np.shape(image)[0]
    image_width = np.shape(image)[1]

    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2

    image = _crop(image, offset_height, offset_width,
                  crop_height, crop_width)
    return image


def _read_video_frame(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))

    image_raw = cv2.imread(filename)

    rows = image_raw.shape[0]
    cols = image_raw.shape[1]
    depth = image_raw.shape[2]
    if cols != 256 or rows != 256:
        print("{} {} {}".format(cols, rows, depth))
    # image rescaled to give in input image aligned with acoustic image
    image = cv2.resize(image_raw, (298, 224), interpolation=cv2.INTER_CUBIC)
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    image_serialized = image.tostring()
    return Image(rows=rows, cols=cols, depth=depth, data=image_serialized)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_raw_dir', help='Synchronized raw files data set root directory', type=str2dir)
    parser.add_argument('out_dir', help='Directory where to store the converted data', type=str2dir)
    parser.add_argument('--modalities', help='Modalities to consider. 0: Audio images. 1: Audio data. 2: Video data.',
                        nargs='*', type=int)
    parsed_args = parser.parse_args()
    root_raw_dir = parsed_args.root_raw_dir
    out_dir = parsed_args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    modalities = parsed_args.modalities
    include_audio_data = modalities is None or 1 in modalities
    include_video_data = modalities is None or 2 in modalities
    with open(root_raw_dir + "/test_list.txt", "r") as outfile:
        test_list = outfile.readlines()
    test_list = [item.replace('\n', '') for item in test_list]
    testing_filename = root_raw_dir + '/test.txt'
    test_tfrecords_list = []
    root_raw_dir = root_raw_dir+"/"+"Dataset"
    data_dirs = sorted(glob.glob('{}/Data/*/'.format(root_raw_dir)))
    #for each fold 0...9
    for data_mat_dir in data_dirs:
        # splitted_data_dir = data_mat_dir.split('/')
        # data_dir_file = str.join('/', splitted_data_dir[:-2])
        #number of tfrecords is equal to number of images and wavs
        video_files = [name for name in os.listdir(data_mat_dir) if name.endswith('.jpg')]
        for image in video_files:
            if image in test_list:
                audio_num = image.split('.jpg')[0]
                filename = '{}/{}.wav'.format(data_mat_dir, audio_num)
                audio_data_raw = _read_wav_audio_data(filename)
                if include_audio_data:
                    audio_data = _read_raw_audio_data(audio_data_raw)
                else:
                    audio_data = None

                if include_video_data:
                    raw_video_files = '{}/{}.jpg'.format(data_mat_dir, audio_num)
                    video_images = _read_video_frame(raw_video_files)
                else:
                    video_images = None

                #read xml
                rows = 256
                cols = 256
                horizontal_scale = 298 / cols
                vertical_scale = 224 / rows

                data_xml_dir = '{}/Annotations/'.format(root_raw_dir)
                root = ET.parse('{}/{}.xml'.format(data_xml_dir, audio_num)).getroot()

                typescene = np.zeros(3, dtype=np.int32)
                xmin = np.zeros(3, dtype=np.int32)
                xmax = np.zeros(3, dtype=np.int32)
                ymin = np.zeros(3, dtype=np.int32)
                ymax = np.zeros(3, dtype=np.int32)

                assert root.find('file_name').text == '{}.jpg'.format(audio_num)

                num_p = -1

                for member in root.findall('person'):
                    num_p += 1
                    bndbox = member.find('bbox')
                    if bndbox.find('type').text == 'object':
                        typescene[num_p] = 1
                    elif bndbox.find('type').text == 'ambient sound':
                        typescene[num_p] = 0
                        print('ambient')
                    else:
                        print(bndbox.find('type').text)
                    xmin[num_p] = bndbox.find('xmin').text
                    xmin[num_p] = int(np.round(xmin[num_p] * horizontal_scale))
                    ymin[num_p] = bndbox.find('ymin').text
                    ymin[num_p] = int(np.round(ymin[num_p] * vertical_scale))
                    xmax[num_p] = bndbox.find('xmax').text
                    xmax[num_p] = int(np.round(xmax[num_p] * horizontal_scale))
                    ymax[num_p] = bndbox.find('ymax').text
                    ymax[num_p] = int(np.round(ymax[num_p] * vertical_scale))
                #number of people
                num_p += 1
                print(xmin.shape)
                out_filename = '{}/{}.tfrecord'.format(out_dir, audio_num)
                print('{} - Writing {}'.format(datetime.now(), out_filename))
                test_tfrecords_list.append(out_filename)
                with tf.python_io.TFRecordWriter(out_filename, options=tf.python_io.TFRecordOptions(
                        compression_type=tf.python_io.TFRecordCompressionType.GZIP)) as writer:
                    # Store audio and video data properties as context features, assuming all sequences are the same size
                    feature = {}
                    if include_audio_data:
                        feature.update({
                            'audio_data/mics': _int64_feature(audio_data.mics),
                            'audio_data/samples': _int64_feature(audio_data.samples)
                        })
                    if include_video_data:
                        feature.update({
                            'video/height': _int64_feature(video_images.rows),
                            'video/width': _int64_feature(video_images.cols),
                            'video/depth': _int64_feature(video_images.depth),
                        })

                    feature_list = {'xmin': tf.train.FeatureList(
                                feature=[_bytes_feature(xmin.tostring())]),
                        'xmax': tf.train.FeatureList(
                                feature=[_bytes_feature(xmax.tostring())]),
                        'ymin': tf.train.FeatureList(
                                feature=[_bytes_feature(ymin.tostring())]),
                        'ymax': tf.train.FeatureList(
                                feature=[_bytes_feature(ymax.tostring())]),
                        'typescene': tf.train.FeatureList(
                                feature=[_bytes_feature(typescene.tostring())])}

                    if include_audio_data:
                        feature_list.update({
                            'audio/data': tf.train.FeatureList(
                                feature=[_bytes_feature(audio_data.data)])
                        })
                    if include_video_data:
                        feature_list.update({
                            'video/image': tf.train.FeatureList(
                                feature=[_bytes_feature(video_images.data)])
                        })
                    context = tf.train.Features(feature=feature)
                    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
                    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                    writer.write(sequence_example.SerializeToString())

    with open(testing_filename, "w") as listfile:
        for t in test_tfrecords_list:
            listfile.write(t +'\n')

    #check numbers in two lists
    t_num = []
    for t in test_tfrecords_list:
         t_num.append(t.split('/')[-1].split('.tfrecord')[0])
    t_num.sort()
    t_num2 = []
    for t in test_list:
         t_num2.append(t.split('.jpg')[0])
    t_num2.sort()
    assert t_num2 == t_num