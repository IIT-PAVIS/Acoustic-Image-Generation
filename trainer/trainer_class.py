from datetime import datetime
from logger.logger import Logger
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from models.base import buildAccuracy
flags = tf.app.flags
FLAGS = flags.FLAGS
_FRAMES_PER_SECOND = 12


class Trainer(object):

    def __init__(self, model, display_freq=1,
                 learning_rate=0.0001, num_classes=14, num_epochs=1, nr_frames=12, temporal_pooling=False):

        self.model = model
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling
        self.shape = [self.model.height, self.model.width, self.model.channels]

    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.acoustic, self.mfcc, self.video, self.labels = self._retrieve_batch(next_batch)
        # self.mfcc = self.mfcc - tf.reduce_min(self.mfcc, axis=[1], keep_dims=True)
        # self.mfcc = self.mfcc/tf.reduce_max(self.mfcc, axis=[1], keep_dims=True)
        if FLAGS.mfccmap:
            mfccmap = tf.reshape(self.mfcc, (-1, 12, 1, 12))
            mfccmap = tf.tile(mfccmap, (1, 1, 36 * 48, 1))
            mfccmap = tf.reshape(mfccmap, (-1, 36, 48, 12))
            considered_modality = mfccmap
        else:
            considered_modality = self.acoustic
        self.model._build_model(considered_modality)
        expanded_shape = [-1, 12, self.num_classes]
        self.logits = tf.reduce_mean(tf.reshape(self.model.output, shape=expanded_shape), axis=1)
        self.cross_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels,
            logits=self.logits,
            scope='cross_loss'
        )
        self.loss = tf.losses.get_total_loss()
        # Define accuracy
        self.accuracy = buildAccuracy(self.logits, self.labels)
        self.global_step = tf.train.create_global_step()
        var_list = slim.get_variables(self.model.scope + '/')
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        with tf.device('/gpu:0'):
            # Compute the gradients for acoustic variables.
            # self.train_op_0 = self.optimizer2.minimize(loss=self.loss,
            #                                            var_list=var_list,
            #                                            global_step=self.global_step)
            # Compute the gradients for acoustic variables.
            grads_and_vars = self.optimizer2.compute_gradients(self.loss, var_list)
            # Ask the optimizer to apply the gradients.
            self.train_op_0 = self.optimizer2.apply_gradients(grads_and_vars, global_step=self.global_step)
        # Initialize model saver
        self.saver = tf.train.Saver(max_to_keep=5)
        return iterat

    def _get_optimizer_variables(self, optimizer, variables):

        optimizer_vars = [optimizer.get_slot(var, name)
                          for name in optimizer.get_slot_names() for var in variables if var is not None]

        optimizer_vars.extend(list(optimizer._get_beta_accumulators()))

        return optimizer_vars

    def _init_model(self, session):

        if FLAGS.init_checkpoint is not None:
            # Restore model
            print('{}: {} - Initializing session'.format(datetime.now(), FLAGS.exp_name))
            #
            session.run(tf.global_variables_initializer())
            var_list1 = slim.get_variables(self.model_encoder_acoustic.scope + '/')
            var_list2 = slim.get_variables(self.model_encoder_images.scope + '/')
            var_list = var_list2 + var_list1
            # else:
            #     var_list = slim.get_model_variables(self.model.scope)
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
        elif FLAGS.restore_checkpoint is not None:
            # Restore session from checkpoint
            self._restore_model(session)
        else:
            # Initialize all variables
            print('{}: {} - Initializing full model'.format(datetime.now(), FLAGS.exp_name))
            session.run(tf.global_variables_initializer())
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

    def _restore_model(self, session):

        # Restore model
        print('{}: {} - Restoring session'.format(datetime.now(), FLAGS.exp_name))
        #
        session.run(tf.global_variables_initializer())
        var_list_ac = slim.get_variables(self.model.scope + '/')
        var_list = var_list_ac
        # else:
        #     var_list = slim.get_model_variables(self.model.scope)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(session, FLAGS.restore_checkpoint)
        print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

    def train(self, train_data=None, valid_data=None):

        # Assert training and validation sets are not None
        assert train_data is not None
        assert valid_data is not None
        # Instantiate logger
        self.logger = Logger('{}/{}'.format(FLAGS.tensorboard, FLAGS.exp_name))
        # Add the variables we train to the summary
        # for var in self.model.train_vars:
        #     self.logger.log_histogram(var.name, var)

        # # Disable image logging
        # self.logger.log_image('input', self.model.network['input'])
        # self.logger.log_sound('input', self.model.network['input'])
        train_iterat = self._build_functions(train_data)
        eval_iterat = valid_data.data.make_initializable_iterator()
        # Add the losses to summary
        self.logger.log_scalar('cross_entropy_loss', self.cross_loss)
        self.logger.log_scalar('train_loss', self.loss)

        # Add the accuracy to the summary
        self.logger.log_scalar('train_accuracy', self.accuracy)
        # Merge all summaries together
        self.logger.merge_summary()

        # Start training session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as session:
            train_handle = session.run(train_iterat.string_handle())
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint
            self._init_model(session)

            # Add the model graph to TensorBoard
            self.logger.write_graph(session.graph)
            self._save_checkpoint(session, 'random')
            start_epoch = int(tf.train.global_step(session, self.global_step))
            best_epoch = -1
            best_accuracy = -1.0
            best_loss = 10000
            # For each epoch
            step2 = 0
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                step = 0
                # Initialize iterator over the training set
                session.run(train_iterat.initializer)
                # For each mini-batch
                while True:
                    try:

                        # Forward batch through the network
                        train_loss, train_accuracy, train_summary, _ = session.run(
                            [self.loss, self.accuracy, self.logger.summary_op, self.train_op_0],
                            feed_dict={self.handle: train_handle,
                                       self.model.network['keep_prob']: 0.5,
                                       self.model.network['is_training']: 1})

                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            print('{}: {} - Iteration: [{:3}]\t Training_Loss: {:6f}\t Training_Accuracy: {:6f}'.format(
                                datetime.now(), FLAGS.exp_name, step, train_loss, train_accuracy))

                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))

                        # Update counters and stats
                        step += 1
                        step2 += 1
                    except tf.errors.OutOfRangeError:
                        break

                session.run(eval_iterat.initializer)
                # Evaluate model on validation set
                total_loss, total_accuracy = self._evaluate(session, 'validation', evaluation_handle)
                print('{}: {} - Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                        FLAGS.exp_name,
                                                                                                        epoch,
                                                                                                        total_loss,
                                                                                                        total_accuracy))

                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss),
                    tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)
                self.logger.flush_writer()
                # if accuracy or loss decrease save model
                if total_accuracy > best_accuracy or (total_accuracy == best_accuracy and total_loss <= best_loss):
                    best_epoch = epoch
                    best_accuracy = total_accuracy
                    best_loss = total_loss
                    # Save model
                    self._save_checkpoint(session, epoch)
                    with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/model.txt", "w") as outfile:
                        outfile.write(
                            '{}: {}\nBest Epoch: {}\nValidation_Loss: {:6f}\nValidation_Accuracy: {:6f}\n'.format(
                                datetime.now(),
                                FLAGS.exp_name,
                                best_epoch,
                                best_loss,
                                best_accuracy))
            print('{}: {} - Best Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                         FLAGS.exp_name,
                                                                                                         best_epoch,
                                                                                                         best_loss,
                                                                                                         best_accuracy))

    def _save_checkpoint(self, session, epoch):

        checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
        model_name = 'epoch_{}.ckpt'.format(epoch)
        print('{}: {} - Saving model to {}/{}'.format(datetime.now(), FLAGS.exp_name, checkpoint_dir, model_name))

        self.saver.save(session, '{}/{}'.format(checkpoint_dir, model_name))

    def _valid(self, session, evaluation_handle):
        return self._evaluate(session, 'validation', evaluation_handle)

    def _evaluate(self, session, mod, eval_handle):

        # Initialize counters and stats
        loss_sum = 0
        accuracy_sum = 0
        data_set_size = 0
        label = []
        pred = []
        # For each mini-batch
        while True:
            try:
                # Compute batch loss and accuracy
                one_hot_labels, logits, batch_loss, batch_accuracy = session.run(
                    [self.labels, self.logits, self.loss, self.accuracy],
                    feed_dict={self.handle: eval_handle,
                               self.model.network['keep_prob']: 1.0,
                               self.model.network['is_training']: 0})
                # Update counters
                data_set_size += one_hot_labels.shape[0]
                loss_sum += batch_loss * one_hot_labels.shape[0]
                accuracy_sum += batch_accuracy * one_hot_labels.shape[0]

            except tf.errors.OutOfRangeError:
                break

        total_loss = loss_sum / data_set_size
        total_accuracy = accuracy_sum / data_set_size
        #if mod == 'test':
        #   self.plot_confusion_matrix(pred, label)
        return total_loss, total_accuracy

    def _retrieve_batch(self, next_batch):

        if FLAGS.model == 'DualCamNet':
            mfcc = tf.reshape(next_batch[1], shape=[-1, 12])
            images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
            acoustic = tf.reshape(next_batch[0], shape=[-1, 12, 36, 48, 12])
            labels = tf.reshape(next_batch[3], shape=[-1, self.num_classes])
        else:
            raise ValueError('Unknown model type')

        return acoustic, mfcc, images, labels

    def test(self, test_data=None):
        name_folder = str.join('/', FLAGS.restore_checkpoint.split('/')[:-1])
        # Assert testing set is not None
        assert test_data is not None
        eval_iterat = self._build_functions(test_data)

        # Start training session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint if given
            self._restore_model(session)
            session.run(eval_iterat.initializer)
            # Evaluate model over the testing set
            test_loss, test_accuracy = self._evaluate(session, 'test', evaluation_handle)
        if FLAGS.mfccmap:
            with open('{}'.format(name_folder) + "/test_accuracy_mfccmap_{}.txt".format(
                    FLAGS.restore_checkpoint.split('/')[-1].split('.')[0].split('_')[1]), "w") as outfile:
                outfile.write('{}: {} - Testing_Loss: {:6f}\nTesting_Accuracy: {:6f}'.format(datetime.now(),
                                                                                             FLAGS.exp_name,
                                                                                             test_loss,
                                                                                             test_accuracy))
        else:
            with open('{}'.format(name_folder) + "/test_accuracy_{}.txt".format(
                    FLAGS.restore_checkpoint.split('/')[-1].split('.')[0].split('_')[1]), "w") as outfile:
                outfile.write('{}: {} - Testing_Loss: {:6f}\nTesting_Accuracy: {:6f}'.format(datetime.now(),
                                                                                             FLAGS.exp_name,
                                                                                             test_loss,
                                                                                             test_accuracy))
        print('{}: {} - Testing_Loss: {:6f}\nTesting_Accuracy: {:6f}'.format(datetime.now(),
                                                                             FLAGS.exp_name,
                                                                             test_loss,
                                                                             test_accuracy))

        return test_loss

    def plot_confusion_matrix(self, pred, label, normalize=True,
                              title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        counter = 0
        cmap = plt.cm.Blues
        cm = confusion_matrix(label, pred)
        percentage2 = label.shape[0]
        for i in range(percentage2):
            if (pred[i] == label[i]):
                counter += 1

        perc = counter / float(percentage2)
        print(perc)
        # classes = ['Clapping', 'Snapping fingers', 'Speaking', 'Whistling', 'Playing kendama', 'Clicking', 'Typing',
        #            'Knocking', 'Hammering', 'Peanut breaking', 'Paper ripping', 'Plastic crumpling', 'Paper shaking',
        #            'Stick dropping']
        classes = ['Train', 'Boat', 'Drone', 'Fountain', 'Drill',
                   'Razor', 'Hair dryer', 'Vacuumcleaner', 'Cart', 'Traffic']
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        # cmap = plt.cm.get_cmap('Blues')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + '/confusion_matrix.png')
