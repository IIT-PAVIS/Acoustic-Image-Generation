import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
flags = tf.app.flags
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('flickr', 0, 'On which dataset')
flags.DEFINE_string('datatype', '', 'music or outdoor or old')
FLAGS = flags.FLAGS

'''Given all IoU files compute area under the curve'''

def main(_):

    plotdecodeimages()

def plotdecodeimages():

    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    if FLAGS.flickr:
        name = '{}_{}_{}_{}'.format('UNet', 'test', 'AcousticFrames{}'.format(FLAGS.datatype), name)
    else:
        name = '{}_{}_{}_{}'.format('UNet', 'testing', 'Acoustictry{}'.format(FLAGS.datatype), name)
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name])
    value = np.zeros(11)
    threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(threshold)):
        with open('{}'.format(data_dir) + "/intersection_{}_accuracy.txt".format(threshold[i]), "r") as outfile:
            t = outfile.read()
            value[i] = t.split(' ')[1]
    value = value[::-1]
    threshold = threshold[::-1]
    plt.plot(threshold, value)
    outImage_path = '{}/auc.png'.format(data_dir, 'testing')
    plt.savefig(outImage_path)
    auc = metrics.auc(threshold, value)
    print('area {:6f}'.format(auc))
    with open('{}'.format(data_dir) + "/area.txt", "w") as outfile:
        outfile.write('area {:6f}'.format(auc))


if __name__ == '__main__':
    tf.app.run()
