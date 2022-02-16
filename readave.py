# Load the Pandas libraries with alias 'pd'
import pandas as pd
import argparse
import os
import numpy as np
import youtube_dl
# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)

'''YouTube videos of AVE dataset'''

parser = argparse.ArgumentParser()
parser.add_argument('csv', help='CSV', type=str)
parser.add_argument('out_dir', help='Directory where to store the converted data', type=str)
parsed_args = parser.parse_args()
csv = parsed_args.csv
outdoor = 0
data = pd.read_csv(csv, sep='&')
out_dir = parsed_args.out_dir
folder = str.join('/', out_dir .split('/')[:-2])
# Preview the first 5 lines of the loaded data
data.head()
res2 = data
names = res2['Category'].unique()
values = np.arange(len(names))
dictionar = dict(zip(names, values))
classesold = ''
d = 0
for index, row in res2.iterrows():
   url = row['VideoID']
   start = row['StartTime']
   end = row['EndTime']
   cl = row['Category']
   classes = dictionar[cl]

   if classes != classesold:
       # data from 0
       d = 0
       classesold = classes
   else:
       # change data folder of 1
       d = d + 1
       #after 3 videos don't save same category
       if d > 7:
           continue
   out_data_dir = '{}/class_{}/data_{:0>3d}/'.format(out_dir, classes, d)
   if not os.path.exists(out_data_dir):
       os.makedirs(out_data_dir)
       os.makedirs(out_data_dir + '/video/')
       os.makedirs(out_data_dir + '/audio/')

   with open('{}/{}'.format(out_data_dir, "/seconds.txt"), "w") as outfile:
       outfile.write('{}:{}\n'.format(start, end))

   video2 = "{}/{}.mp4".format(folder, url)
   os.system("ffmpeg -i {} -vf fps=12 {}video/I_%06d.bmp".format(video2, out_data_dir))
   #read original audio sometimes higher sampling rate
   os.system("ffmpeg -i {} {}output_audio.wav".format(video2, out_data_dir))
   #convert to mono 12*1024  16 bit
   os.system("ffmpeg -i {}output_audio.wav -ar 12288 -acodec pcm_s16le -f wav -ac 1 {}audio/output_audio2.wav".format(out_data_dir, out_data_dir))

# os.system("ls") to use terminal command use os
