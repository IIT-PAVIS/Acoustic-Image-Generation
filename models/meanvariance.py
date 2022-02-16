# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

def mean_std(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
      "Batch Normalization: Accelerating Deep Network Training by Reducing
      Internal Covariate Shift"
      Sergey Ioffe, Christian Szegedy
    Can be used as a normalizer function for conv2d and fully_connected.
    Args:
      inputs: a tensor of size `[batch_size, height, width, channels]`
              or `[batch_size, channels]`.
      decay: decay for the moving average.
      center: If True, subtract `beta`. If False, `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: small float added to variance to avoid dividing by zero.
      activation_fn: Optional activation function.
      updates_collections: collections to collect the update ops for computation.
        If None, a control dependency would be added to make sure the updates are
        computed.
      is_training: whether or not the layer is in training mode. In training mode
        it would accumulate the statistics of the moments into `moving_mean` and
        `moving_variance` using an exponential moving average with the given
        `decay`. When it is not in training mode then it would use the values of
        the `moving_mean` and the `moving_variance`.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional collections for the variables.
      outputs_collections: collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_op_scope`.
    Returns:
      a tensor representing the output of the operation.
    """
    with variable_scope.variable_op_scope([inputs],
                                          scope, 'BatchNorm', reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        dtype = inputs.dtype.base_dtype
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]

        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(
            variables_collections, 'moving_mean')
        moving_mean = variables.model_variable(
            'moving_mean',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.zeros_initializer,
            trainable=False,
            collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(
            variables_collections, 'moving_variance')
        moving_variance = variables.model_variable(
            'moving_variance',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.ones_initializer,
            trainable=False,
            collections=moving_variance_collections)
        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = nn.moments(inputs, axis, shift=moving_mean)
            # Update the moving_mean and moving_variance moments.
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            if updates_collections is None:
                # Make sure the updates are computed here.
                with ops.control_dependencies([update_moving_mean,
                                               update_moving_variance]):
                    outputs = nn.batch_normalization(
                        inputs, mean, variance)
            else:
                # Collect the updates to be computed later.
                ops.add_to_collections(updates_collections, update_moving_mean)
                ops.add_to_collections(updates_collections, update_moving_variance)
                outputs = nn.batch_normalization(
                    inputs, mean, variance, epsilon)
        else:
            outputs = nn.batch_normalization(
                inputs, moving_mean, moving_variance, epsilon)
        outputs.set_shape(inputs.get_shape())
        if activation_fn:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

