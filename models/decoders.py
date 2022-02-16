import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict

flags = tf.app.flags
FLAGS = flags.FLAGS

class DecoderVideo(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'DecoderVideo'
        self.num_frames = num_frames
        self.height = input_shape[0]
        self.embedding = embedding

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, is_training=True, keep_prob=0.5, scope='DecoderVideo'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'DecoderVideo', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer()):
                        #stride = 2 double padding valid increase of kernel -1
                        net = slim.fully_connected(inputs, 36*48, activation_fn=tf.nn.relu)
                        net = slim.fully_connected(net, 224*298, activation_fn=tf.nn.relu)
                        net = tf.reshape(net, shape=[-1, 224, 298, 1])
                        net = slim.conv2d(net, 8, [3, 3], scope='conv_0')
                        net = slim.conv2d(net, 64, [3, 3], scope='conv_1')
                        net = slim.conv2d(net, 512, [3, 3], scope='conv_2')
                        net = slim.conv2d(net, 128, [3, 3], scope='conv_3')
                        net = slim.conv2d(net, 64, [3, 3], scope='conv_4')
                        net = slim.conv2d(net, 32, [3, 3], scope='conv_5')
                        net = slim.conv2d(net, 16, [3, 3], scope='conv_6')
                        net = slim.conv2d(net, 8, [3, 3], scope='conv_7', activation_fn=None)
                        net = slim.conv2d(net, 3, [3, 3], scope='conv_8', activation_fn=tf.nn.sigmoid)
        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


    def _build_model(self, acoustic_images):
        """
        Builds the hybrid model using slim and base functions.
        """
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, is_training=is_training,
                                                                             scope=self.scope)

        end_points.update(dualcam_net_end_points)


        self.output = dualcam_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)

class DualCamHybridModelDecoderEnergy(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'DualCamNetDecoderEnergy'
        self.num_frames = num_frames
        self.height = input_shape[0]
        self.embedding = embedding

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, is_training=True, keep_prob=0.5, scope='DualCamNetDecoderEnergy'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'DualCamNetDecoderEnergy', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu,
                                stride=1, weights_initializer=tf.contrib.layers.xavier_initializer()):
                    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu):
                        #stride = 2 double padding valid increase of kernel -1
                        net = slim.fully_connected(inputs, 12*16, activation_fn=tf.nn.relu)
                        net = slim.fully_connected(net, 36*48, activation_fn=tf.nn.relu)
                        net = tf.reshape(net, shape=[-1, 36, 48, 1])
                        net = slim.conv2d(net, 64, [5, 5],  scope='conv_0')
                        net = slim.conv2d(net, 32, [5, 5],  scope='conv_1')
                        net = slim.conv2d(net, 16, [5, 5],  scope='conv_2')
                        net = slim.conv2d(net, 8, [3, 3],  scope='conv_3')
                        net = slim.conv2d(net, 4, [3, 3], scope='conv_4')
                        net = slim.conv2d(net, 2, [3, 3], scope='conv_5')
                        net = slim.conv2d(net, 1, [3, 3], scope='conv_6')
        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


    def _build_model(self, acoustic_images):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, is_training=is_training,
                                                                             scope=self.scope)

        end_points.update(dualcam_net_end_points)


        self.output = dualcam_net_output  # shared_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)

class DecoderAudio(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'DecoderAudio'
        self.num_frames = num_frames
        self.height = input_shape[0]
        self.embedding = embedding

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, is_training=True, keep_prob=0.5, scope='DecoderAudio'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'DecoderAudio', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu,
                                stride=1, weights_initializer=tf.contrib.layers.xavier_initializer()):
                    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu):
                        #stride = 2 double padding valid increase of kernel -1
                        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu)
                        net = slim.fully_connected(net, 12288, activation_fn=tf.nn.relu)
                        net = tf.reshape(net, shape=[-1, 12288, 1, 1])
                        net = slim.layers.conv2d(net, 128, [1024, 1], [1, 1], scope='conv1')
                        net = slim.layers.conv2d(net, 64, [512, 1], [1, 1], scope='conv2')
                        net = slim.layers.conv2d(net, 32, [128, 1], [1, 1], scope='conv3')
                        net = slim.layers.conv2d(net, 16, [32, 1], [1, 1], scope='conv4')
                        net = slim.layers.conv2d(net, 8, [16, 1], [1, 1], scope='conv5')
                        net = slim.layers.conv2d(net, 4, [3, 1], [1, 1], scope='conv6')
                        net = slim.layers.conv2d(net, 1, [1, 1], [1, 1], scope='conv7')
        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


    def _build_model(self, acoustic_images):
        """
        Builds the hybrid model using slim and base functions.
        """
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, is_training=is_training,
                                                                             scope=self.scope)

        end_points.update(dualcam_net_end_points)


        self.output = dualcam_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)