import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
#https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_input.py
class VGGish(object):

    def __init__(self, input_shape=None, num_classes=10):
        self.scope = 'vggish'

        self.num_classes = num_classes

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        # Restore all the model layers
        variables_to_restore = slim.get_variables(self.scope)  # + slim.get_variables('shared_net')

        # Initialization operation of the pre-trained weights
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)

        # Load the pre-trained weights
        init_fn(session)

    def _build_network(self, inputs, scope='VGGish'):
        """
        Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
        """

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=INIT_STDDEV),
                            biases_initializer=tf.zeros_initializer(),
                            activation_fn=tf.nn.relu,
                            trainable=False), \
             slim.arg_scope([slim.conv2d],
                            kernel_size=[3, 3], stride=1, padding='SAME'), \
             slim.arg_scope([slim.max_pool2d],
                            kernel_size=[2, 2], stride=2, padding='SAME'), \
             tf.variable_scope('vggish'):
            # Input: a batch of 2-D log-mel-spectrogram patches.
            features = inputs
                # tf.placeholder(
                # tf.float32, shape=(None, NUM_FRAMES, NUM_BANDS),
                # name='input_features')
            # Reshape to 4-D so that we can convolve a batch with conv2d().
            net = tf.reshape(features, [-1, NUM_FRAMES, NUM_BANDS, 1])
            # net = tf.reshape(features, [-1, 99, 257, 1])

            # The VGG stack of alternating convolutions and max-pools.
            net = slim.conv2d(net, 64, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            net = slim.conv2d(net, 128, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
            net = slim.max_pool2d(net, scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
            net = slim.max_pool2d(net, scope='pool4')

            # Flatten before entering fully-connected layers
            net = slim.flatten(net)
            net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
            # The embedding layer.
            # net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc2')
            net = tf.reshape(net, [-1, 1, 1, 4096])
            return tf.identity(net, name='embedding')

    def _build_model(self, melspectrum):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # end_points = OrderedDict({
        #     'input': melspectrum,
        #     'is_training': is_training,
        #     'keep_prob': keep_prob
        # })

        output = self._build_network(melspectrum, scope=self.scope)

        # end_points.update(dualcam_net_end_points)

        self.output = output  # shared_net_output
        # self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)
