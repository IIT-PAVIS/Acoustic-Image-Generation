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

'''Write a file with YouTube links'''

parser = argparse.ArgumentParser()
parser.add_argument('csv', help='CSV', type=str)
parser.add_argument('out_dir', help='Directory where to store the converted data', type=str)
parsed_args = parser.parse_args()
csv = parsed_args.csv

data = pd.read_csv(csv)
out_dir = parsed_args.out_dir
testing_filename = 'videolista.txt'
# Preview the first 5 lines of the loaded data
data.head()
data0 = data[data['class'].str.contains('train wagon')]
data1 = data[data['class'].str.contains('motorboat')]
data3 = data[data['class'].str.contains('waterfall')]
data5 = data[data['class'].str.contains('razor')]
data6 = data[data['class'].str.contains('hair dryer')]
data7 = data[data['class'].str.contains('vacuum cleaner')]
data9 = data[data['class'].str.contains('car passing by')]
res = pd.concat([data0, data1, data3, data5, data6, data7, data9])
#res2 contains video to consider
res2 = res[res['set'].str.contains('test')]
names = res2['class'].unique()
values = np.asarray([0, 1, 3, 5, 6, 7, 9])
dictionar = dict(zip(names, values))
classesold = ''
try:
    f_testing = open(out_dir + testing_filename, 'w')
except:
    print('error while opening file: ')
    print_exc()

for index, row in res2.iterrows():
   url = row['url']
   s = row['seconds']
   cl = row['class']
   classes = dictionar[cl]
   ydl_opts = {
       'format': 'bestaudio/best',
       'outtmpl': 'tmp/%(id)s.%(ext)s',
       'noplaylist': True,
       'quiet': True,
       'prefer_ffmpeg': True,
       'audioformat': 'wav',
       'forceduration': True
   }
   sID = url
   l = 0
   with youtube_dl.YoutubeDL(ydl_opts) as ydl:
       try:
           dictMeta = ydl.extract_info(
           "https://www.youtube.com/watch?v={sID}".format(sID=sID), download=False)
           length = dictMeta['duration']
       except:
           continue
   min = length // 60
   second = length % 60
   #check length is less than three minutes before download
   if min < 3:
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
           print("https://www.youtube.com/watch?v={}".format(url) + '\n')
           f_testing.write("https://www.youtube.com/watch?v={}".format(url) + '\n')
           #after 7 videos don't save same category
f_testing.close()
