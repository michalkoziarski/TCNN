import os
import logging
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from tqdm import tqdm
from trainer import MODELS_PATH


class Network(ABC):
    def __init__(self, name, output_shape, input_shape=None, inputs=None, include_top=True):
        assert inputs is not None or input_shape is not None

        self.name = name
        self.output_shape = output_shape

        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))

        self.outputs = self.inputs
        self.include_top = include_top
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

    def predict(self, dataset, session=None, restore=False, verbose=False):
        session_passed = session is not None

        if not session_passed:
            session = tf.Session()

        if restore:
            self.restore(session)

        probabilities = np.full((dataset.length, self.output_shape[0]), np.nan)

        iterator = range(0, dataset.length, dataset.batch_size)

        if verbose:
            logging.info('Evaluating the model...')

            iterator = tqdm(iterator)

        for position in iterator:
            batch, _ = dataset.batch()
            batch_probabilities = tf.nn.softmax(self.outputs).eval(feed_dict={self.inputs: batch}, session=session)
            probabilities[position:(position + dataset.batch_size)] = batch_probabilities

        if not session_passed:
            session.close()

        return probabilities

    def accuracy(self, dataset, session=None, restore=False, verbose=False):
        predictions = self.predict(dataset, session, restore, verbose)

        return np.mean(np.argmax(predictions, 1) == dataset.labels)

    def restore(self, session, model_path=None):
        if model_path is None:
            model_path = os.path.join(MODELS_PATH, self.name)

        checkpoint = tf.train.get_checkpoint_state(model_path)

        assert checkpoint
        assert checkpoint.model_checkpoint_path

        tf.train.Saver().restore(session, checkpoint.model_checkpoint_path)

    @abstractmethod
    def setup(self):
        pass


class C3DNetwork(Network):
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
            fully_connected('fully_connected_7', [2048, 2048], tf.nn.relu)

        if self.include_top:
            self.fully_connected('fully_connected_8', [2048, self.output_shape[0]])


class C2DNetwork(Network):
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
            fully_connected('fully_connected_7', [2048, 2048], tf.nn.relu)

        if self.include_top:
            self.fully_connected('fully_connected_8', [2048, self.output_shape[0]])


class MultiStreamNetwork(Network):
    def __init__(self, name, output_shape, input_shape, stream_types=('C3D', 'C2D_0', 'C2D_1', 'C2D_2'),
                 stream_names=None):
        assert len(input_shape) == len(stream_types)

        self.name = name
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.flat_shape = np.prod(self.input_shape[0])
        self.inputs = tf.placeholder(tf.float32, shape=[None, len(stream_types), self.flat_shape])
        self.stream_types = stream_types
        self.stream_names = stream_names
        self.stream_inputs = []
        self.streams = []
        self.setup()

    def setup(self):
        for i in range(len(self.stream_types)):
            self.stream_inputs.append(tf.reshape(self.inputs[:, i], [-1] + list(self.input_shape[i])))

            if self.stream_types[i].startswith('C3D'):
                stream_class = C3DNetwork
            else:
                stream_class = C2DNetwork

            if self.stream_names is None:
                stream_name = '%s_%s_Stream_%d' % (self.name, self.stream_types[i], i)
            else:
                stream_name = self.stream_names[i]

            self.streams.append(stream_class(stream_name, None, inputs=self.stream_inputs[i], include_top=False))

            for value in tf.get_collection('%s_weight_decay' % stream_name):
                tf.add_to_collection('%s_weight_decay' % self.name, value)

        self.outputs = tf.concat([stream.outputs for stream in self.streams], 1)

        self.fully_connected('fully_connected_8', [-1, self.output_shape[0]])
