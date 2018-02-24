import os
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from model import MultiStreamNetwork


MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
LOGS_PATH = os.path.join(os.path.dirname(__file__), 'logs')


class Trainer:
    def __init__(self, network, train_set, epochs, learning_rate, evaluation_step=1, validation_set=None,
                 early_stopping=False, use_pretrained_streams=False, verbose=False):
        self.network = network
        self.train_set = train_set
        self.epochs = epochs
        self.evaluation_step = evaluation_step
        self.validation_set = validation_set
        self.early_stopping = early_stopping
        self.use_pretrained_streams = use_pretrained_streams
        self.verbose = verbose
        self.global_step = tf.get_variable('%s_global_step' % network.name, [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.latest_accuracy = tf.get_variable('%s_latest_accuracy' % network.name, [],
                                               initializer=tf.constant_initializer(-np.inf),
                                               trainable=False)
        self.ground_truth = tf.placeholder(tf.int64, shape=[train_set.batch_size])
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                       logits=network.outputs))
        self.weight_decay_loss = tf.reduce_mean(tf.get_collection('%s_weight_decay' % self.network.name))
        self.total_loss = self.base_loss + self.weight_decay_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
        self.saver = tf.train.Saver()

        tf.summary.scalar('base_loss', self.base_loss)
        tf.summary.scalar('weight_decay_loss', self.weight_decay_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        if validation_set is not None:
            self.validation_accuracy = tf.placeholder(tf.float32)

            tf.summary.scalar('validation_accuracy', self.validation_accuracy)

        self.summary_step = tf.summary.merge_all()

        self.model_path = os.path.join(MODELS_PATH, network.name)
        self.checkpoint_path = os.path.join(self.model_path, '%s.ckpt' % network.name)
        self.log_path = os.path.join(LOGS_PATH, network.name)

        self.summary_writer = tf.summary.FileWriter(self.log_path)

        for path in [MODELS_PATH, LOGS_PATH, self.model_path, self.log_path]:
            if not os.path.exists(path):
                os.mkdir(path)

    def train(self):
        with tf.Session() as session:
            checkpoint = tf.train.get_checkpoint_state(self.model_path)
            session.run(tf.global_variables_initializer())

            if checkpoint and checkpoint.model_checkpoint_path:
                if self.verbose:
                    logging.info('Restoring the model...')

                self.saver.restore(session, checkpoint.model_checkpoint_path)
            elif self.use_pretrained_streams:
                assert isinstance(self.network, MultiStreamNetwork)

                if self.verbose:
                    logging.info('Loading pretrained streams...')

                self.network.load_pretrained_streams(session)

            batches_processed = tf.train.global_step(session, self.global_step)
            batches_per_epoch = int(self.train_set.length / self.train_set.batch_size)
            epochs_processed = int(batches_processed / batches_per_epoch)

            for epoch in range(epochs_processed, self.epochs):
                if self.verbose:
                    logging.info('Shuffling training set...')

                self.train_set.shuffle()

                batch_iterator = range(0, batches_per_epoch - 1)

                if self.verbose:
                    logging.info('Processing epoch #%d...' % (epoch + 1))
                    batch_iterator = tqdm(batch_iterator)

                for _ in batch_iterator:
                    inputs, outputs = self.train_set.batch()
                    feed_dict = {self.network.inputs: inputs, self.ground_truth: outputs}
                    session.run([self.train_step], feed_dict=feed_dict)

                inputs, outputs = self.train_set.batch()
                feed_dict = {self.network.inputs: inputs, self.ground_truth: outputs}

                if self.validation_set is not None:
                    validation_accuracy = self.network.accuracy(self.validation_set, session, verbose=self.verbose)

                    if self.verbose:
                        logging.info('Observed %.4f validation accuracy.' % validation_accuracy)

                    if self.early_stopping and session.run(self.latest_accuracy) > validation_accuracy:
                        session.run(self.latest_accuracy.assign(validation_accuracy))

                        if self.verbose:
                            logging.info('Stopping the training early.')

                        break

                    session.run(self.latest_accuracy.assign(validation_accuracy))

                    feed_dict[self.validation_accuracy] = validation_accuracy

                _, summary = session.run([self.train_step, self.summary_step], feed_dict=feed_dict)

                self.saver.save(session, self.checkpoint_path)
                self.summary_writer.add_summary(summary, epochs_processed)
