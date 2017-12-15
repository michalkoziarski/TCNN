import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs
        self.outputs = self.inputs
        self.setup()

    def convolution(self, layer_name, shape, weight_decay=0.0005, stride=1, padding='SAME', kind='2d'):
        assert kind in ['2d', '3d']

        weights = self._variable('%s_weights' % layer_name, shape, weight_decay)
        biases = self._variable('%s_biases' % layer_name, [shape[-1]])

        if kind == '2d':
            self.outputs = tf.nn.conv2d(self.outputs, weights, [1, stride, stride, 1], padding=padding)
        else:
            self.outputs = tf.nn.conv3d(self.outputs, weights, [1, stride, stride, stride, 1], padding=padding)

        self.outputs = tf.nn.relu(tf.nn.bias_add(self.outputs, biases))

        return self

    def convolution2d(self, layer_name, shape, weight_decay=0.0005, stride=1, padding='SAME'):
        return self.convolution(layer_name, shape, weight_decay, stride, padding, '2d')

    def convolution3d(self, layer_name, shape, weight_decay=0.0005, stride=1, padding='SAME'):
        return self.convolution(layer_name, shape, weight_decay, stride, padding, '3d')

    def pooling2d(self, layer_name, k, padding='SAME'):
        if type(k) is int:
            k = [1, k, k, 1]

        self.outputs = tf.nn.max_pool(self.outputs, ksize=k, strides=k, padding=padding, name=layer_name)

        return self

    def pooling3d(self, layer_name, k, padding='SAME'):
        if type(k) is int:
            k = [1, k, k, k, 1]

        self.outputs = tf.nn.max_pool3d(self.outputs, ksize=k, strides=k, padding=padding, name=layer_name)

        return self

    def fully_connected(self, layer_name, shape, weight_decay=0.0005, activation=None):
        if shape[0] == -1:
            shape[0] = int(np.prod(self.outputs.get_shape()[1:]))

        weights = self._variable('%s_weights' % layer_name, shape, weight_decay)
        biases = self._variable('%s_biases' % layer_name, [shape[-1]])

        self.outputs = tf.nn.bias_add(tf.matmul(self.outputs, weights), biases)

        if activation is not None:
            self.outputs = activation(self.outputs)

        return self

    def flatten(self):
        flattened_shape = int(np.prod(self.outputs.get_shape()[1:]))

        self.outputs = tf.reshape(self.outputs, [-1, flattened_shape])

        return self

    def _variable(self, variable_name, shape, weight_decay=None, initializer=None):
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()

        variable = tf.get_variable(variable_name, shape, initializer=initializer)

        if weight_decay is not None:
            tf.add_to_collection('%s_weight_decay' % self.name, tf.nn.l2_loss(variable) * weight_decay)

        return variable

    @abstractmethod
    def setup(self):
        pass
