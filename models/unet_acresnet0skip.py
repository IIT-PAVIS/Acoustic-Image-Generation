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

class UNetAc(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=False):

        self.scope = 'UNetAcRes'
        self.num_frames = num_frames
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        #if true, no vae use auto encoder
        self.embedding = embedding

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, resnetfeature, is_training=True, keep_prob=0.5, weight_decay=1e-6, scope='UNetAcRes'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
        with tf.variable_scope(scope, 'UNetAcRes', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            #net = X / 127.5 - 1
            conv1, pool1 = self.conv_conv_pool(inputs, [128, 128], is_training, weight_decay, name="1", strides=(3, 3))
            conv2_0 = self.conv_conv_pool(
                pool1, [133, 133], is_training, weight_decay, name="2", pool=False)
            conv2 = conv2_0 - tf.reduce_min(conv2_0, axis=[1, 2, 3], keep_dims=True)
            conv2 = conv2/tf.reduce_max(conv2, axis=[1, 2, 3], keep_dims=True)
            resnetfeature = resnetfeature - tf.reduce_min(resnetfeature, axis=[1, 2, 3], keep_dims=True)
            resnetfeature = resnetfeature/tf.reduce_max(resnetfeature, axis=[1, 2, 3], keep_dims=True)
            #concatenate normalized feature maps
            conv2 = tf.concat((conv2, resnetfeature), axis=-1)

            if self.embedding:
                guessed_z = tf.layers.conv2d(conv2, 150, (12, 16), padding='VALID', name='mean')
                guessed_z = tf.reshape(guessed_z, (-1, 150))
                # std = tf.nn.softplus(tf.layers.conv2d(conv2, 150, (12, 16), padding='VALID', name='std'))
                # std = tf.reshape(std, (-1, 150))
                # samples = tf.random_normal([tf.shape(std)[0], tf.shape(std)[1]], 0, 1, dtype=tf.float32)
                # guessed_z = mean + (std * samples)
                guessed_z = guessed_z - tf.reduce_min(guessed_z, axis=1, keep_dims=True)
                guessed_z = guessed_z / tf.reduce_max(guessed_z, axis=1, keep_dims=True)
            else:
                mean = tf.layers.conv2d(conv2, 150, (12, 16), padding='VALID', name='mean')
                mean = tf.reshape(mean, (-1, 150))
                std = tf.nn.softplus(tf.layers.conv2d(conv2, 150, (12, 16), padding='VALID', name='std'))
                std = tf.reshape(std, (-1, 150))
                samples = tf.random_normal([tf.shape(std)[0], tf.shape(std)[1]], 0, 1, dtype=tf.float32)
                guessed_z = mean + (std * samples)

            net = tf.layers.dense(guessed_z, 12 * 16 * 12, activation=tf.nn.relu)
            net = tf.reshape(net, (-1, 12, 16, 12))
            net = tf.layers.conv2d(net, 133, (3, 3), activation=tf.nn.relu, padding='same')
            # up0 = tf.concat([net, conv2_0], axis=-1, name="concat_{}".format("0"))
            conv4 = self.conv_conv_pool(net, [128, 128], is_training, weight_decay, name="4", pool=False)
            conv5 = self.conv_conv_pool(conv4, [128, 128], is_training, weight_decay, name="5", pool=False)
            up1 = self.upconv_conc(conv5, 128, weight_decay, name="1", strides=3)
            conv6 = self.conv_conv_pool(up1, [128, 128], is_training, weight_decay, name="6", pool=False)
            conv7 = self.conv_conv_pool(conv6, [64, 64], is_training, weight_decay, name="7", pool=False)
            net = tf.layers.conv2d(
                conv7,
                12, (3, 3),
                name='final',
                activation=tf.nn.sigmoid,
                padding='same')

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
        end_points['features'] = conv2
        if self.embedding:
            return guessed_z, net, end_points
        else:
            return mean, std, net, end_points


    def _build_model(self, acoustic_images, resnetfeature):
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
        if not self.embedding:
            mean, std, dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, resnetfeature, is_training=is_training,
                                                                                 scope=self.scope)
        else:
            mean, dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images, resnetfeature,
                                                                                        is_training=is_training,
                                                                                        scope=self.scope)

        end_points.update(dualcam_net_end_points)

        self.mean = mean
        if not self.embedding:
            self.std = std
        self.output = dualcam_net_output  # shared_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)

    def conv_conv_pool(self, input_,
                       n_filters,
                       is_training,
                       weight_decay,
                       name,
                       pool=True,
                       activation=tf.nn.relu, padding='same', filters=(3,3), strides=(2,2)):
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
                    kernel_regularizer=None,
                    name="conv_{}".format(i + 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
                # net = tf.layers.batch_normalization(
                #     net, training=is_training, name="bn_{}".format(i + 1))
                net = activation(net, name="relu{}_{}".format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.conv2d(
                    net,
                    F, filters, strides=strides,
                    activation=None,
                    padding=padding,
                    kernel_regularizer=None,
                    name="pool_{}".format(i + 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            # pool = tf.layers.batch_normalization(
            #         pool, training=is_training, name="bn_pool_{}".format(i + 1))
            pool = activation(pool, name="relu_pool{}_{}".format(name, i + 1))

            return net, pool

    def upconv_conc(self, inputA, n_filter, weight_decay, name, kernel_size=2, strides=2):
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
            kernel_regularizer=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="upsample_{}".format(name))