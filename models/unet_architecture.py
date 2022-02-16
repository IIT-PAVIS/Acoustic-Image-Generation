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
        # self.output, self.network = self._build_model()

        # self.train_vars = slim.get_trainable_variables(self.scope)

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
            conv1, pool1 = self.conv_conv_pool(inputs, [8, 8], is_training, weight_decay, name="1")
            conv2, pool2 = self.conv_conv_pool(pool1, [32, 32], is_training, weight_decay, name="2", padding='valid', filters=(2, 3))
            conv3, pool3 = self.conv_conv_pool(pool2, [32, 32], is_training, weight_decay, name="3")
            conv4, pool4 = self.conv_conv_pool(pool3, [64, 64], is_training, weight_decay, name="4", padding='valid', filters=(2, 3))
            conv5 = self.conv_conv_pool(
                pool4, [128, 128], is_training, weight_decay, name="5", pool=False)

            # embedding = math_ops.reduce_mean(conv5, [1, 2], keepdims=True)
            mean = tf.layers.conv2d(conv5, 128, (14, 18), padding='VALID', name='mean')
            mean = tf.reshape(mean, (-1, 128))
            variance = tf.layers.conv2d(conv5, 128, (14, 18), padding='VALID', name='variance')
            variance = tf.reshape(variance, (-1, 128))
            samples = tf.random_normal([tf.shape(variance)[0], tf.shape(variance)[1]], 0, 1, dtype=tf.float32)
            guessed_z = mean + (variance * samples)

            net = tf.layers.dense(guessed_z, 14 * 18, activation=tf.nn.relu)
            net = tf.reshape(net, (-1, 14, 18, 1))
            net = tf.layers.conv2d(net, 128, (3, 3), activation=tf.nn.relu, padding='same')

            up6 = self.upconv_concat(net, conv4, 64, weight_decay, name="6", kernel_size=[2, 3])
            conv6 = self.conv_conv_pool(up6, [64, 64], is_training, weight_decay, name="6", pool=False)

            up7 = self.upconv_concat(conv6, conv3, 32, weight_decay, name="7")
            conv7 = self.conv_conv_pool(up7, [32, 32], is_training, weight_decay, name="7", pool=False)

            up8 = self.upconv_concat(conv7, conv2, 32, weight_decay, name="8", kernel_size=[2, 3])
            conv8 = self.conv_conv_pool(up8, [32, 32], is_training, weight_decay, name="8", pool=False)

            up9 = self.upconv_concat(conv8, conv1, 8, weight_decay, name="9")
            conv9 = self.conv_conv_pool(up9, [8, 8], is_training, weight_decay, name="9", pool=False)

            net = tf.layers.conv2d(
                conv9,
                3, (1, 1),
                name='final',
                activation=tf.nn.sigmoid,
                padding='same')

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
        end_points['features'] = conv5
        return mean, variance, net, end_points


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

        mean, variance, dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, is_training=is_training,
                                                                             scope=self.scope)

        end_points.update(dualcam_net_end_points)

        self.mean = mean
        self.variance = variance
        self.output = dualcam_net_output  # shared_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope + '/')

    def conv_conv_pool(self, input_,
                       n_filters,
                       is_training,
                       weight_decay,
                       name,
                       pool=True,
                       activation=tf.nn.relu, padding='same', filters=(3,3)):
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
                    F, (3, 3),
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
                    F, filters, strides=(2, 2),
                    activation=None,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name="pool_{}".format(i + 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool = tf.layers.batch_normalization(
                    pool, training=is_training, name="bn_pool_{}".format(i + 1))
            pool = activation(pool, name="relu_pool{}_{}".format(name, i + 1))

            return net, pool

    def upconv_concat(self, inputA, input_B, n_filter, weight_decay, name, kernel_size=2):
        """Upsample `inputA` and concat with `input_B`
        Args:
            input_A (4-D Tensor): (N, H, W, C)
            input_B (4-D Tensor): (N, 2*H, 2*H, C2)
            name (str): name of the concat operation
        Returns:
            output (4-D Tensor): (N, 2*H, 2*W, C + C2)
        """
        up_conv = self.upconv_2D(inputA, n_filter, weight_decay, name, kernel_size=kernel_size)

        return tf.concat(
            [up_conv, input_B], axis=-1, name="concat_{}".format(name))

    def upconv_2D(self, tensor, n_filter, weight_decay, name, kernel_size=2):
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
            strides=2,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="upsample_{}".format(name))
