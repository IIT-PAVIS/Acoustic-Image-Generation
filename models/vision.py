from collections import OrderedDict
from tensorflow.contrib.slim.nets import resnet_v1 as resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models import resnet50

class ResNet50Model(object):
    def __init__(self, input_shape=None, num_classes=None):
        self.scope = 'resnet_v1_50'

        self.num_classes = num_classes

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

        # self.output, self.network = self._build_model()


    def init_model(self, session, checkpoint_file):
        """
        Initializes ResNet-50 network parameters using slim.
        """

        # Restore only the layers up to logits (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['logits', '/conv_map'])
        # Initialization operation of the pre-trained weights
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)

        # Initialization operation from scratch for the new "logits" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        logits_variables = slim.get_model_variables(self.scope + '/logits')
        logits_init_op = tf.variables_initializer(logits_variables)
        conv_map_variables = slim.get_model_variables(self.scope + '/conv_map')
        conv_map_init_op = tf.variables_initializer(conv_map_variables)
        # Load the pre-trained weights
        init_fn(session)

        # Initialize the new logits layer
        session.run(logits_init_op)
        session.run(conv_map_init_op)


    def _build_model(self, visual_images):
        """
        Builds a ResNet-50 network using slim.
        """

        # visual_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='visual_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with slim.arg_scope(resnet.resnet_arg_scope(weight_decay=5e-4)):
            output, network = resnet50.resnet_v1_50(visual_images, num_classes=self.num_classes,
                                                    is_training=is_training, global_pool=False)

        # output = tf.squeeze(output, [1, 2])

        network.update({
            'input': visual_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        self.output = output
        self.network = network
        self.train_vars2 = slim.get_trainable_variables(self.scope + '/block') + slim.get_trainable_variables(
            self.scope + '/conv1')
        self.train_vars = slim.get_trainable_variables(self.scope + '/logits') + slim.get_trainable_variables(
            self.scope + '/conv_map')
        # return output, network
        # self.train_vars = slim.get_trainable_variables(self.scope + '/logits') + slim.get_trainable_variables(
        #     self.scope + '/conv_map')