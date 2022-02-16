# Load the Pandas libraries with alias 'pd'
import pandas as pd
import argparse
import os
import numpy as np
import youtube_dl
from traceback import print_exc
# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)

'''YouTube videos of VGG sound dataset'''

parser = argparse.ArgumentParser()
parser.add_argument('csv', help='CSV', type=str)
parser.add_argument('out_dir', help='Directory where to store the converted data', type=str)
parsed_args = parser.parse_args()
csv = parsed_args.csv
outdoor = 0
data = pd.read_csv(csv)
out_dir = parsed_args.out_dir
# Preview the first 5 lines of the loaded data
data.head()
#save url
testing_filename = 'videolista.txt'
num_videos = 20
if outdoor:
    data0 = data[data['class'].str.contains('train wagon')]
    data1 = data[data['class'].str.contains('motorboat')]
    data3 = data[data['class'].str.contains('waterfall')]
    data5 = data[data['class'].str.contains('razor')]
    data6 = data[data['class'].str.contains('hair dryer')]
    data7 = data[data['class'].str.contains('vacuum cleaner')]
    data9 = data[data['class'].str.contains('car passing by')]
    res = pd.concat([data0, data1, data3, data5, data6, data7, data9])#
    #res2 contains video to consider
    res2 = res[res['set'].str.contains('test')]
    names = res2['class'].unique()
    values = np.asarray([0, 1, 3,5, 6, 7, 9])
    dictionar = dict(zip(names, values))
else:
    data0 = data[data['class'].str.contains('clapping')]
    data1 = data[data['class'].str.contains('people finger snapping')]
    data2 = data[data['class'].str.contains('male speech, man speaking')]
    data3 = data[data['class'].str.contains('people whistling')]
    data5 = data[data['class'].str.contains('clicking')]
    data6 = data[data['class'].str.contains('typing on computer keyboard')]
    data8 = data[data['class'].str.contains('hammering')]
    data10 = data[data['class'].str.contains('ripping paper')]
    data11 = data[data['class'].str.contains('plastic')]
    res = pd.concat([data0, data1, data2, data3, data5, data6, data8, data10, data11])
    #res2 contains video to consider
    res2 = res[res['set'].str.contains('test')]
    names = res2['class'].unique()
    values = np.asarray([0, 1, 2, 3, 5, 6, 8, 10, 11])
    dictionar = dict(zip(names, values))
classesold = ''

try:
    f_testing = open(out_dir + testing_filename, 'w')
except:
    print('error while opening file: ')
    print_exc()

d = 0
for index, row in res2.iterrows():
   url = row['url']
   s = row['seconds']
   cl = row['class']
   classes = dictionar[cl]
   # ydl_opts = {
   #     'format': 'bestaudio/best',
   #     'outtmpl': 'tmp/%(id)s.%(ext)s',
   #     'noplaylist': True,
   #     'quiet': True,
   #     'prefer_ffmpeg': True,
   #     'audioformat': 'wav',
   #     'forceduration': True
   # }
   # sID = url
   # l = 0
   # with youtube_dl.YoutubeDL(ydl_opts) as ydl:
   #     try:
   #         dictMeta = ydl.extract_info(
   #         "https://www.youtube.com/watch?v={sID}".format(sID=sID), download=False)
   #         length = dictMeta['duration']
   #     except:
   #         continue
   # min = length // 60
   # second = length % 60
   #check length is less than three minutes before download
   # if min < 3:
   if classes != classesold:
       # data from 0
       d = 0
       classesold = classes
       print(cl + '\n')
       f_testing.write(cl + '\n')
       print("https://www.youtube.com/watch?v={}".format(url) + '\n')
       f_testing.write("https://www.youtube.com/watch?v={}".format(url) + '\n')
   else:
       # change data folder of 1
       d = d + 1
       if d > num_videos:
           continue
       print("https://www.youtube.com/watch?v={}".format(url) + '\n')
       f_testing.write("https://www.youtube.com/watch?v={}".format(url) + '\n')
       #after 20 videos don't save same category

   out_data_dir = '{}/class_{}/data_{:0>3d}/'.format(out_dir, classes, d)
   if not os.path.exists(out_data_dir):
       os.makedirs(out_data_dir)
       os.makedirs(out_data_dir + '/video/')
       os.makedirs(out_data_dir + '/audio/')


   os.system("ffmpeg -y $(youtube-dl -g 'https://www.youtube.com/watch?v={}' | sed \"s/.*/-ss {} -i &/\")  -t 10 -c copy {}out.mkv".format(url, s, out_data_dir))
   # os.system("youtube-dl -o {}all.mp4 https://www.youtube.com/watch?v={}".format(out_data_dir, url))
   video = os.popen("ls {} -p | grep -v / ".format(out_data_dir)).read().split('\n')[0]
   if len(video) > 0:
       name = video.split('.')[0]
       ending = video.split('.')[1]
       video2 = "{}2.{}".format(name, ending)
       # start from the end of start sec experimental
       # os.system("ffmpeg -y -ss 5 -t 10 -i {}{} -strict experimental -movflags faststart {}{}".format(out_data_dir, video, out_data_dir, video2))
       # os.system("rm {}{}".format(out_data_dir, video))
       os.system("ffmpeg -i {}{} -vf fps=12 {}video/I_%06d.bmp".format(out_data_dir, video, out_data_dir))
       frames = os.popen("ls {}video -p | grep -v / ".format(out_data_dir)).read().split('\n')[0]
       if frames == '':
           d=d-1
           continue
       #read original audio sometimes higher sampling rate
       os.system("ffmpeg -i {}{} {}output_audio.wav".format(out_data_dir, video, out_data_dir))
       #convert to mono 12*1024  16 bit
       os.system("ffmpeg -i {}output_audio.wav -ar 12288 -acodec pcm_s16le -f wav -ac 1 {}audio/output_audio2.wav".format(out_data_dir, out_data_dir))
       with open('{}/{}'.format(out_data_dir, "url.txt"), "w") as outfile:
           outfile.write("https://www.youtube.com/watch?v={}".format(url))
   else:
       d=d-1

f_testing.close()
# os.system("ls") to use terminal command use os
