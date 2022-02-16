import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict

flags = tf.app.flags
FLAGS = flags.FLAGS

"""Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """

class UNet(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'UNet'

        self.num_frames = num_frames

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.embedding = embedding
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.end_points = OrderedDict({
            'is_training': self.is_training,
            'keep_prob': self.keep_prob
        })

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, is_training=True, keep_prob=0.5, weight_decay=7e-5, scope='UNet'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'UNet', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            #net = X / 127.5 - 1
            conv1, pool1 = self.conv_conv_pool(inputs, [32, 32], self.is_training, weight_decay, name="1", strides=(3, 3), padding='valid', filters=(3, 3))
            conv2, pool2 = self.conv_conv_pool(pool1, [128, 128], self.is_training, weight_decay, name="2", padding='valid', filters=(3, 3))
            conv3, pool3 = self.conv_conv_pool(pool2, [256, 256], self.is_training, weight_decay, name="3", strides=(3, 3), padding='valid', filters=(2, 3))
            conv5 = self.conv_conv_pool(
                pool3, [512, 512], is_training, weight_decay, name="5", pool=False)

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
        end_points['features'] = conv5
        self.end_points.update(end_points)
        return conv5

    def _build_network2(self, f, is_training=True, keep_prob=0.5, weight_decay=7e-5, scope='UNet'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'UNet', [f]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            mean = tf.layers.conv2d(f, 1024, (12, 16), padding='VALID', name='mean')
            mean = tf.reshape(mean, (-1, 1024))
            std = tf.nn.softplus(tf.layers.conv2d(f, 1024, (12, 16), padding='VALID', name='std'))
            std = tf.reshape(std, (-1, 1024))
            samples = tf.random_normal([tf.shape(std)[0], tf.shape(std)[1]], 0, 1, dtype=tf.float32)
            guessed_z = mean + (std * samples)

            net = tf.layers.dense(guessed_z, 12 * 16 * 50, activation=tf.nn.relu)
            net = tf.reshape(net, (-1, 12, 16, 50))
            net = tf.layers.conv2d(net, 512, (3, 3), activation=tf.nn.relu, padding='same')

            up6 = self.upconv_concat(net, 256, weight_decay, name="6", kernel_size=[3, 4], strides=3)
            conv6 = self.conv_conv_pool(up6, [256, 256], is_training, weight_decay, name="6", pool=False)
            conv7 = self.conv_conv_pool(conv6, [256, 256], is_training, weight_decay, name="7", pool=False)

            up8 = self.upconv_concat(conv7, 128, weight_decay, name="8", kernel_size=[4, 3])
            conv8 = self.conv_conv_pool(up8, [128, 128], is_training, weight_decay, name="8", pool=False)
            conv9 = self.conv_conv_pool(conv8, [128, 128], is_training, weight_decay, name="9", pool=False)

            up10 = self.upconv_concat(conv9, 32, weight_decay, name="10", kernel_size=[5, 4], strides=3)
            conv10 = self.conv_conv_pool(up10, [32, 32], is_training, weight_decay, name="10", pool=False)
            conv11 = self.conv_conv_pool(conv10, [32, 32], is_training, weight_decay, name="11", pool=False)

            net = tf.layers.conv2d(
                conv11,
                3, (1, 1),
                name='final',
                activation=tf.nn.sigmoid,
                padding='same')

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
        return mean, std, net, end_points


    def _build_model(self, f):
        """
        Builds the hybrid model using slim and base functions.
        """

        mean, std, dualcam_net_output, dualcam_net_end_points = self._build_network2(f, is_training=self.is_training,
                                                                             scope=self.scope)

        self.end_points.update(dualcam_net_end_points)
        self.mean = mean
        self.std = std
        self.output = dualcam_net_output  # shared_net_output
        self.network = self.end_points
        self.train_vars = slim.get_trainable_variables(self.scope + '/')

    def conv_conv_pool(self, input_,
                       n_filters,
                       is_training,
                       weight_decay,
                       name,
                       pool=True,
                       activation=tf.nn.relu, padding='same', filtersconv=(3,3), filters=(3,3), strides=(2,2)):
        """{Conv -> BN -> RELU}x2 -> {Pool, optional}
        Args:
            input_ (4-D Tensor): (batch_size, H, W, C)
            n_filters (list): number of filters [int, int]
            training (1-D Tensor): Boolean Tensor
            name (str): name postfix
            pool (bool): If True, MaxPool2D
            activation: Activaion functions
        Returns:
            net: output of the Convolution operations
            pool (optional): output of the max pooling operations
        """
        net = input_

        with tf.variable_scope("layer{}".format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(
                    net,
                    F, filtersconv,
                    activation=None,
                    padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name="conv_{}".format(i + 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(
                    net, training=is_training, name="bn_{}".format(i + 1))
                net = activation(net, name="relu{}_{}".format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.conv2d(
                    net,
                    F, filters, strides=strides,
                    activation=None,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name="pool_{}".format(i + 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool = tf.layers.batch_normalization(
                    pool, training=is_training, name="bn_pool_{}".format(i + 1))
            pool = activation(pool, name="relu_pool{}_{}".format(name, i + 1))

            return net, pool

    def upconv_concat(self, inputA, n_filter, weight_decay, name, kernel_size=2, strides=2):
        """Upsample `inputA` and concat with `input_B`
        Args:
            input_A (4-D Tensor): (N, H, W, C)
            input_B (4-D Tensor): (N, 2*H, 2*H, C2)
            name (str): name of the concat operation
        Returns:
            output (4-D Tensor): (N, 2*H, 2*W, C + C2)
        """
        up_conv = self.upconv_2D(inputA, n_filter, weight_decay, name, kernel_size=kernel_size, strides=strides)

        return up_conv

    def upconv_2D(self, tensor, n_filter, weight_decay, name, kernel_size=2, strides=2):
        """Up Convolution `tensor` by 2 times
        Args:
            tensor (4-D Tensor): (N, H, W, C)
            n_filter (int): Filter Size
            name (str): name of upsampling operations
        Returns:
            output (4-D Tensor): (N, 2 * H, 2 * W, C)
        """

        return tf.layers.conv2d_transpose(
            tensor,
            filters=n_filter,
            kernel_size=kernel_size,
            strides=strides,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="upsample_{}".format(name))
