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

    def fully_connected(self, layer_name, shape, activation=None, weight_decay=0.0005):
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

        variable = tf.get_variable('%s_%s' % (self.name, variable_name), shape, initializer=initializer)

        if weight_decay is not None:
            tf.add_to_collection('%s_weight_decay' % self.name, tf.nn.l2_loss(variable) * weight_decay)

        return variable

    def predict(self, inputs, batch_size, session=None):
        session_passed = session is not None

        if not session_passed:
            session = tf.Session()

        predictions = np.full(len(inputs), np.nan)
        position = 0

        while position < len(inputs):
            batch = inputs[position:(position + batch_size)]
            batch_predictions = tf.argmax(self.outputs, 1).eval(feed_dict={self.inputs: batch}, session=session)
            predictions[position:(position + batch_size)] = batch_predictions
            position += batch_size

        if not session_passed:
            session.close()

        return predictions

    def accuracy(self, inputs, ground_truth, batch_size, session=None):
        predictions = self.predict(inputs, batch_size, session)

        return np.mean(predictions == ground_truth)

    @abstractmethod
    def setup(self):
        pass


class C3DNetwork(Network):
    def __init__(self, input_shape, output_shape, name='C3D'):
        self.input_shape = input_shape
        self.output_shape = output_shape

        inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))

        super().__init__(name, inputs)

    def setup(self):
        self.convolution3d('convolution_1', [3, 3, 3, 3, 64]).\
            pooling3d('pooling_1', [1, 1, 2, 2, 1]).\
            convolution3d('convolution_2', [3, 3, 3, 64, 128]).\
            pooling3d('pooling_2', [1, 2, 2, 2, 1]).\
            convolution3d('convolution_3_1', [3, 3, 3, 128, 256]).\
            convolution3d('convolution_3_2', [3, 3, 3, 256, 256]).\
            pooling3d('pooling_3', [1, 2, 2, 2, 1]).\
            convolution3d('convolution_4_1', [3, 3, 3, 256, 512]).\
            convolution3d('convolution_4_2', [3, 3, 3, 512, 512]).\
            pooling3d('pooling_4', [1, 2, 2, 2, 1]).\
            convolution3d('convolution_5_1', [3, 3, 3, 512, 512]).\
            convolution3d('convolution_5_2', [3, 3, 3, 512, 512]).\
            pooling3d('pooling_5', [1, 2, 2, 2, 1]).\
            flatten().\
            fully_connected('fully_connected_6', [-1, 2048], tf.nn.relu).\
            fully_connected('fully_connected_7', [2048, 2048], tf.nn.relu).\
            fully_connected('fully_connected_8', [2048, self.output_shape[0]])


class C2DNetwork(Network):
    def __init__(self, input_shape, output_shape, name='C2D'):
        self.input_shape = input_shape
        self.output_shape = output_shape

        inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))

        super().__init__(name, inputs)

    def setup(self):
        self.convolution2d('convolution_1', [3, 3, 3, 64]).\
            pooling2d('pooling_1', [1, 1, 4, 1]).\
            convolution2d('convolution_2', [3, 3, 64, 128]).\
            pooling2d('pooling_2', [1, 2, 4, 1]).\
            convolution2d('convolution_3_1', [3, 3, 128, 256]).\
            convolution2d('convolution_3_2', [3, 3, 256, 256]).\
            pooling2d('pooling_3', [1, 2, 4, 1]).\
            convolution2d('convolution_4_1', [3, 3, 256, 512]).\
            convolution2d('convolution_4_2', [3, 3, 512, 512]).\
            pooling2d('pooling_4', [1, 2, 4, 1]).\
            convolution2d('convolution_5_1', [3, 3, 512, 512]).\
            convolution2d('convolution_5_2', [3, 3, 512, 512]).\
            pooling2d('pooling_5', [1, 2, 4, 1]).\
            flatten().\
            fully_connected('fully_connected_6', [-1, 2048], tf.nn.relu).\
            fully_connected('fully_connected_7', [2048, 2048], tf.nn.relu).\
            fully_connected('fully_connected_8', [2048, self.output_shape[0]])
