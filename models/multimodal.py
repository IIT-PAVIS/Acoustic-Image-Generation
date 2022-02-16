import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict

class AssociatorVideoAc(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'AssociatorVideoAc'
        self.num_frames = num_frames
        self.height = input_shape

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, mean, std, scope='AssociatorVideoAc'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'AssociatorVideoAc', [mean, std]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            #from 1024 to 128
            net = tf.layers.dense(mean, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 256, activation=tf.nn.relu)
            net = tf.layers.dense(net, 256, activation=tf.nn.relu)
            net = tf.layers.dense(net, 150, activation=tf.nn.relu)
            mean = tf.layers.dense(net, 150, activation=None)
            mean = tf.reshape(mean, (-1, 150))

            net2 = tf.layers.dense(std, 512, activation=tf.nn.relu)
            net2 = tf.layers.dense(net2, 512, activation=tf.nn.relu)
            net2 = tf.layers.dense(net2, 256, activation=tf.nn.relu)
            net2 = tf.layers.dense(net2, 256, activation=tf.nn.relu)
            net2 = tf.layers.dense(net2, 150, activation=tf.nn.relu)
            std = tf.nn.softplus(tf.layers.dense(net2, 150, activation=None))
            std = tf.reshape(std, (-1, 150))

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return mean, std, end_points

    def _build_model(self, mean, std):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': mean,
            'input2': std
            # 'is_training': is_training,
            # 'keep_prob': keep_prob
        })

        mean, std, end_points2 = self._build_network(mean, std, scope=self.scope)

        end_points.update(end_points2)

        self.std = std
        self.mean = mean
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)

class AssociatorAudioAc(object):

    def __init__(self, input_shape=None, num_frames=12, embedding=True):
        self.scope = 'AssociatorAudioAc'
        self.num_frames = num_frames
        self.height = input_shape

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, mean, std, scope='AssociatorAudioAc'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'AssociatorAudioAc', [mean, std]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            #from 256 to 128
            net = tf.layers.dense(mean, 256, activation=tf.nn.relu)
            net = tf.layers.dense(net, 256, activation=tf.nn.relu)
            mean = tf.layers.dense(net, 150, activation=None)
            mean = tf.reshape(mean, (-1, 150))

            net2 = tf.layers.dense(std, 256, activation=tf.nn.relu)
            net2 = tf.layers.dense(net2, 256, activation=tf.nn.relu)
            std = tf.nn.softplus(tf.layers.dense(net2, 150, activation=None))
            std = tf.reshape(std, (-1, 150))

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return mean, std, end_points

    def _build_model(self, mean, std):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': mean,
            'input2': std
            # 'is_training': is_training,
            # 'keep_prob': keep_prob
        })

        mean, std, end_points2 = self._build_network(mean, std, scope=self.scope)

        end_points.update(end_points2)

        self.std = std
        self.mean = mean
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)

class AssociatorAudio(object):

    def __init__(self, input_shape=None):
        self.scope = 'AssociatorAudio'
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputs, is_training, scope='AssociatorAudio', weight_decay=8e-5):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'AssociatorAudio', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            conv1, pool1 = self.conv_conv_pool(inputs, [16, 16], is_training, weight_decay, name="1", padding='valid',
                                               filters=(3, 3))
            conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], is_training, weight_decay, name="2")
            conv3, pool3 = self.conv_conv_pool(pool2, [64, 64], is_training, weight_decay, name="3")
            conv4, pool4 = self.conv_conv_pool(pool3, [128, 128], is_training, weight_decay, name="4")
            conv5 = self.conv_conv_pool(
                pool4, [128, 128], is_training, weight_decay, name="5", pool=False)

            # embedding = math_ops.reduce_mean(conv5, [1, 2], keepdims=True)
            mean = tf.layers.conv2d(conv5, 150, (12, 16), padding='VALID', name='mean')
            mean = tf.reshape(mean, (-1, 150))
            std = tf.nn.softplus(tf.layers.conv2d(conv5, 150, (12, 16), padding='VALID', name='std'))
            std = tf.reshape(std, (-1, 150))

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return mean, std, end_points

    def _build_model(self, inputs):
        """
        Builds the hybrid model using slim and base functions.
        """

        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': inputs,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        mean, std, end_points2 = self._build_network(inputs, is_training, scope=self.scope)

        end_points.update(end_points2)

        self.std = std
        self.mean = mean
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)
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

    def upconv_concat(self, inputA, n_filter, weight_decay, name, kernel_size=2):
        """Upsample `inputA` and concat with `input_B`
        Args:
            input_A (4-D Tensor): (N, H, W, C)
            input_B (4-D Tensor): (N, 2*H, 2*H, C2)
            name (str): name of the concat operation
        Returns:
            output (4-D Tensor): (N, 2*H, 2*W, C + C2)
        """
        up_conv = self.upconv_2D(inputA, n_filter, weight_decay, name, kernel_size=kernel_size)

        return up_conv

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


class Jointmvae(object):

    def __init__(self, input_shape=None):
        self.scope = 'Jointmvae'

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputac, inputvideo, inputaudio, scope='Jointmvae'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'Jointmvae', [inputac, inputvideo, inputaudio]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            #from 12x16x(128, 512, 128) concatenated to each 12x16x128, 1024, 256 features
            input = tf.concat((inputac, inputvideo, inputaudio), axis=-1)
            net = tf.layers.dense(input, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            outputac = tf.layers.dense(net, 133, activation=tf.nn.relu)
            outputvideo = tf.layers.dense(net, 512, activation=tf.nn.relu)
            outputaudio = tf.layers.dense(net, 128, activation=tf.nn.relu)

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return outputac, outputvideo, outputaudio, end_points

    def _build_model(self, inputac, inputvideo, inputaudio):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': inputac,
            'input2': inputvideo,
            'input3': inputaudio
            # 'is_training': is_training,
            # 'keep_prob': keep_prob
        })

        outputac, outputvideo, outputaudio, end_points2 = self._build_network(inputac, inputvideo, inputaudio, scope=self.scope)

        end_points.update(end_points2)

        self.outputac = outputac
        self.outputvideo = outputvideo
        self.outputaudio = outputaudio
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)

class JointTwomvae(object):

    def __init__(self, input_shape=None):
        self.scope = 'JointTwomvae'

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputvideo, inputaudio, scope='JointTwomvae'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'JointTwomvae', [inputvideo, inputaudio]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            #from 12x16x(512, 128) concatenated to each 12x16x128 features
            input = tf.concat((inputvideo, inputaudio), axis=-1)
            net = tf.layers.dense(input, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            outputac = tf.layers.dense(net, 133, activation=tf.nn.relu)

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return outputac, end_points

    def _build_model(self, inputvideo, inputaudio):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input2': inputvideo,
            'input3': inputaudio
            # 'is_training': is_training,
            # 'keep_prob': keep_prob
        })

        outputac, end_points2 = self._build_network(inputvideo, inputaudio, scope=self.scope)

        end_points.update(end_points2)

        self.outputac = outputac
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)

class JointTwomvae2(object):

    def __init__(self, input_shape=None):
        self.scope = 'JointTwomvae2'

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_network(self, inputvideo, inputaudio, scope='JointTwomvae2'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """

        with tf.variable_scope(scope, 'JointTwomvae2', [inputvideo, inputaudio]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            #from 12x16x(512, 128) concatenated to each 12x16x128 features
            input = tf.concat((inputvideo, inputaudio), axis=-1)
            net = tf.layers.dense(input, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            outputac = tf.layers.dense(net, 133, activation=tf.nn.relu)
            outputvideo = tf.layers.dense(net, 512, activation=tf.nn.relu)
            outputaudio = tf.layers.dense(net, 128, activation=tf.nn.relu)

        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

        return outputac, outputvideo, outputaudio, end_points

    def _build_model(self, inputvideo, inputaudio):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input2': inputvideo,
            'input3': inputaudio
            # 'is_training': is_training,
            # 'keep_prob': keep_prob
        })

        outputac, outputvideo, outputaudio, end_points2 = self._build_network(inputvideo, inputaudio, scope=self.scope)

        end_points.update(end_points2)

        self.outputac = outputac
        self.outputvideo = outputvideo
        self.outputaudio = outputaudio
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)
