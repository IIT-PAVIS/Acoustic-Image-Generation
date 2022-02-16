# ''' contare il numero di frame dentro la cartella per trovare
# quanto lungo il video in secondi, salvare un file txt sulla durata
# in secondi del video arrotondata per difetto (int) '''
#
# ''' nella cartella radice (la cartella class) salvare un file che contenga
# la somma delle durate dei video dei file salvati nella cartella data '''

# sys is used for extracting script arguments from bash
import sys
# glob is used to list files in directories using regex
import glob
# os is used to obtain script directory
import argparse
from traceback import print_exc

import numpy as np
from scipy.io import wavfile

FRAMERATE = 12  # number of frames

parser = argparse.ArgumentParser()
parser.add_argument('root_raw_dir', help='Synchronized raw files data set root directory', type=str)
parser.add_argument('out_dir', help='Directory where to store the converted data', type=str)
parsed_args = parser.parse_args()
parsed_args = parser.parse_args()
path = parsed_args.root_raw_dir
path2 = parsed_args.out_dir
tfrecord = 1
# list containing classes' directories
classes_dir = glob.glob(path + '/class_*/')
classes_dir.sort()
# '''files that will contain single video length and sum of each video length
# of the class'''
video_time_filename = 'video_time.txt'
class_time_filename = 'class_time.txt'
testing_filename = 'testing_file.txt'

i = 0

'''Run first with tfrecord=0 with 2 times data folder, then convert_data, then with tfrecords=1 to save lists of tfrecords
  with folder and tfrecords folder  and then use script to create videos for VGG Sound'''

for c in classes_dir:
    # seconds counter for each class
    class_seconds = 0

    # list containing data directory for each class
    data_dir = glob.glob(c + '/data_*/')
    data_dir.sort()
    num_datadir = len(data_dir)
    for d in data_dir:
        testing_list = []
        try:
            save = path2 + '/' + str.join('/', d.split('/')[-3:])
            f_testing = open(save + '/' + 'testing_file.txt', 'w')
        except:
            print('error while opening file: ')
            print_exc()

        video_dir = d + '/video'
        audio_dir = d + '/audio'
        if tfrecord:
            tot_frames = len((glob.glob(d + '/*.tfrecord')))
            images = glob.glob(d + '/*.tfrecord')
            images.sort()
            for b in images:
                testing_list.append(b)
            # using integer division
            video_seconds = tot_frames  # // FRAMERATE
        else:
            tot_frames = len((glob.glob(video_dir + '/*.bmp')))
            images = glob.glob(video_dir + '/*.bmp')
            images.sort()
            for b in images:
                testing_list.append(b)
            # using integer division
            video_seconds = tot_frames // FRAMERATE
            if video_seconds > 0:
                filename = audio_dir + '/output_audio2.wav'
                fs, data = wavfile.read(filename)
                samples = len(data)//(12*1024)
                video_seconds = np.minimum(video_seconds, samples)
                wavfile.write(filename, 12 * 1024, data[0:video_seconds*12*1024])
        for v in testing_list:
            f_testing.write(v + '\n')
        i = i + 1
        # video seconds has o be miminum 2 seconds or divisible for 2
        # if it is 0 we don't add
        # if it is odd we subtract one
        # if video_seconds%2!=0:
        #     video_seconds-=1
        class_seconds = class_seconds + video_seconds

        # exception management
        try:
            fv = open(d + video_time_filename, 'w')
            fv.write('video seconds: {}'.format(video_seconds))
            fv.close()
        except:
            print('error during writing video_time.txt for ' + d)
    try:
        fc = open(c + '/' + class_time_filename, 'w')
        fc.write('class seconds: {}'.format(class_seconds))
        fc.close()
    except:
        print('error during writing class_time.txt for ' + c)