import os
import tensorflow as tf


MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
LOGS_PATH = os.path.join(os.path.dirname(__file__), 'logs')


class Trainer:
    def __init__(self, network, dataset, epochs, learning_rate, evaluation_step=1, validation_set=None, verbose=False):
        self.network = network
        self.dataset = dataset
        self.epochs = epochs
        self.evaluation_step = evaluation_step
        self.validation_set = validation_set
        self.verbose = verbose
        self.global_step = tf.get_variable('%s_global_step' % network.name, [], initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.ground_truth = tf.placeholder(tf.int64, shape=[dataset.batch_size])
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                       logits=network.outputs))
        self.weight_decay_loss = tf.reduce_mean(tf.get_collection('%s_weight_decay' % self.network.name))
        self.total_loss = self.base_loss + self.weight_decay_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
        self.saver = tf.train.Saver()

        tf.summary.scalar('%s: base loss' % network.name, self.base_loss)
        tf.summary.scalar('%s: weight decay loss' % network.name, self.weight_decay_loss)
        tf.summary.scalar('%s: total loss' % network.name, self.total_loss)

        if validation_set is not None:
            self.validation_accuracy = tf.placeholder(tf.float32)

            tf.summary.scalar('%s: validation accuracy' % network.name, self.validation_accuracy)

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
                    print('Restoring the model...')

                self.saver.restore(session, checkpoint.model_checkpoint_path)

            while True:
                batches_processed = tf.train.global_step(session, self.global_step)
                epochs_processed = batches_processed * self.dataset.batch_size / self.dataset.length

                if epochs_processed >= self.epochs:
                    break

                inputs, outputs = self.dataset.batch()
                feed_dict = {self.network.inputs: inputs, self.ground_truth: outputs}

                if (batches_processed * self.dataset.batch_size) % (self.dataset.length * self.evaluation_step) == 0:
                    if self.verbose:
                        print('Processing epoch #%d...' % (epochs_processed + 1))

                    if self.validation_set is not None:
                        validation_accuracy = self.network.accuracy(self.validation_set.videos,
                                                                    self.validation_set.labels,
                                                                    self.validation_set.batch_size, session)
                        feed_dict[self.validation_accuracy] = validation_accuracy

                    _, summary = session.run([self.train_step, self.summary_step], feed_dict=feed_dict)

                    self.saver.save(session, self.checkpoint_path)
                    self.summary_writer.add_summary(summary, epochs_processed)
                else:
                    session.run([self.train_step], feed_dict=feed_dict)
