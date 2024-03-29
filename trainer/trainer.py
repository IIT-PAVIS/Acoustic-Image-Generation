from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

flags = tf.app.flags
FLAGS = flags.FLAGS
_FRAMES_PER_SECOND = 12


class Trainer(object):

    def __init__(self, model, logger=None, display_freq=1,
                 learning_rate=0.0001, num_classes=14, num_epochs=1, nr_frames=12, temporal_pooling=False):

        self.model = model
        self.logger = logger
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling
        self.shape = [self.model.height, self.model.width, self.model.channels]
        self.anneal_function = 'logistic'
        self.x0 = 2500
        self.k = 0.0025

    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        self.steps = tf.placeholder(tf.float32, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.acoustic, self.logenergy, self.mfcc, self.video, self.labels = self._retrieve_batch(next_batch)
        self.logenergy = self.logenergy - tf.reduce_min(self.logenergy, axis=[1, 2], keep_dims=True)
        self.logenergy = self.logenergy / tf.reduce_max(self.logenergy, axis=[1, 2], keep_dims=True)
        # self.mfcc = self.mfcc - tf.reduce_min(self.mfcc, axis=[1, 2], keep_dims=True)
        # self.mfcc = self.mfcc/tf.reduce_max(self.mfcc, axis=[1, 2], keep_dims=True)

        if FLAGS.encoder_type == 'Video':
            considered_modality = self.video
        elif FLAGS.encoder_type == 'Audio':
            considered_modality = self.mfcc
        elif FLAGS.encoder_type == 'Ac':
            considered_modality = self.acoustic
        else:
            considered_modality = self.logenergy

        #UNet
        self.model._build_model(considered_modality)
        self.losslmse = tf.losses.mean_squared_error(considered_modality, self.model.output)
        self.l1 = tf.losses.huber_loss(considered_modality, self.model.output)
        self.latent_loss = 0.5 * tf.reduce_mean(tf.square(self.model.mean) + tf.square(self.model.std)
                                               - tf.log(1e-8 + tf.square(self.model.std)) - 1, 1)
        self.global_step = tf.train.create_global_step()
        # if FLAGS.encoder_type == 'Ac':
        #     self.l1b1 = tf.losses.huber_loss(self.model.c1, self.model.b1)
        #     self.l1b2 = tf.losses.huber_loss(self.model.c2, self.model.b2)
        # offset = tf.range(12)
        # channels = tf.map_fn(lambda o: tf.slice(considered_modality, [0, 0, 0, o], [-1, 36, 48, 1]),
        #                           offset, dtype=considered_modality.dtype)
        # channelstrue = tf.map_fn(lambda o: tf.slice(self.model.output, [0, 0, 0, o], [-1, 36, 48, 1]),
        #                           offset, dtype=self.model.output.dtype)
        # self.l2losses = tf.reduce_mean(tf.pow(channels-channelstrue, 2), axis=[1, 2, 3, 4]) / 2

        self.latent_loss = tf.reduce_mean(self.latent_loss, 0)/1000000
        var_list = self.model.train_vars
        self.loss = self.latent_loss + tf.losses.get_total_loss()

        # Initialize counters and stats


        # Define optimizer
        # before different
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.device('/gpu:0'):
                # Compute the gradients for acoustic variables.
                self.train_op_0 = self.optimizer2.minimize(loss=self.loss,
                                                           var_list=var_list,
                                                           global_step=self.global_step)
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
            # not to have uninitialized value
            # session.run(tf.global_variables_initializer())
            # Initialize global step
            print('{}: {} - Initializing global step'.format(datetime.now(), FLAGS.exp_name))
            session.run(self.global_step.initializer)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            # don't initialize with sgd and momentum
            # Initialize optimizer variables
            print('{}: {} - Initializing optimizer variables'.format(datetime.now(), FLAGS.exp_name))
            optimizer_vars = self._get_optimizer_variables(self.optimizer)
            optimizer_init_op = tf.variables_initializer(optimizer_vars)
            session.run(optimizer_init_op)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            # Initialize model
            print('{}: {} - Initializing model'.format(datetime.now(), FLAGS.exp_name))
            self.model.init_model(session, FLAGS.init_checkpoint)
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
        if FLAGS.mode == "train":
            if FLAGS.model == 'ResNet50':
                to_exclude = [i.name for i in tf.global_variables()
                              if
                              '/Adam' in i.name or 'power' in i.name or '/full1' in i.name or 'conv_map' in i.name or 'logits' in i.name
                              ]
            elif FLAGS.model == 'HearNet':
                to_exclude = [i.name for i in tf.global_variables()
                              if '/Adam' in i.name or 'power' in i.name or 'fc3' in i.name]
            else:  # FLAGS.model == 'DualCamHybridNet':
                to_exclude = [i.name for i in tf.global_variables()
                              if
                              '/Adam' in i.name or 'power' in i.name or '/full1' in i.name]
                # only to finetune dualcamhybridmodel
        else:
            to_exclude = [i.name for i in tf.global_variables()
                          if '/Adam' in i.name or 'power' in i.name]
        var_list = slim.get_variables_to_restore(exclude=to_exclude)
        # else:
        #     var_list = slim.get_model_variables(self.model.scope)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(session, FLAGS.restore_checkpoint)
        print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

    def train(self, train_data=None, valid_data=None):

        # Assert training and validation sets are not None
        assert train_data is not None
        assert valid_data is not None

        # Add the variables we train to the summary
        # for var in self.model.train_vars:
        #     self.logger.log_histogram(var.name, var)

        # # Disable image logging
        # self.logger.log_image('input', self.model.network['input'])
        # self.logger.log_sound('input', self.model.network['input'])
        train_iterat = self._build_functions(train_data)
        eval_iterat = valid_data.data.make_initializable_iterator()
        # Add the losses to summary
        self.logger.log_scalar('l2 loss', self.losslmse)
        self.logger.log_scalar('l1 loss', self.l1)
        self.logger.log_scalar('train_loss', self.loss)
        self.logger.log_scalar('latent_loss', self.latent_loss)
        if FLAGS.encoder_type == 'Video':
            self.logger.log_image('image', self.video)
        elif FLAGS.encoder_type == 'Audio':
            self.logger.log_image('sound', self.mfcc)
        elif FLAGS.encoder_type == 'Ac':
            # for i in range(12):
            #     mfcc1 = tf.squeeze(tf.slice(self.l2losses, [i], [1]))
            #     self.logger.log_scalar('l2_mfcc{}'.format(i), mfcc1)
            mfccelement0 = tf.slice(self.acoustic, [0, 0, 0, 0], [-1, 36, 48, 3])
            mfccelement0 = mfccelement0 - tf.reduce_min(mfccelement0, axis=[1, 2, 3], keep_dims=True)
            mfccelement0 = mfccelement0 / tf.reduce_max(mfccelement0, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('mfcc0', mfccelement0, max_outputs=1)
            mfccelement1 = tf.slice(self.acoustic, [0, 0, 0, 3], [-1, 36, 48, 3])
            mfccelement1 = mfccelement1 - tf.reduce_min(mfccelement1, axis=[1, 2, 3], keep_dims=True)
            mfccelement1 = mfccelement1 / tf.reduce_max(mfccelement1, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('mfcc1', mfccelement1, max_outputs=1)
            mfccelement2 = tf.slice(self.acoustic, [0, 0, 0, 6], [-1, 36, 48, 3])
            mfccelement2 = mfccelement2 - tf.reduce_min(mfccelement2, axis=[1, 2, 3], keep_dims=True)
            mfccelement2 = mfccelement2 / tf.reduce_max(mfccelement2, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('mfcc2', mfccelement2, max_outputs=1)
            mfccelement3 = tf.slice(self.acoustic, [0, 0, 0, 9], [-1, 36, 48, 3])
            mfccelement3 = mfccelement3 - tf.reduce_min(mfccelement3, axis=[1, 2, 3], keep_dims=True)
            mfccelement3 = mfccelement3 / tf.reduce_max(mfccelement3, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('mfcc3', mfccelement3, max_outputs=1)
        else:
            self.logger.log_image('input', self.logenergy)

        #UNet
        if FLAGS.encoder_type == 'Ac':
            # self.logger.log_scalar('l1_b1', self.l1b1)
            # self.logger.log_scalar('l1_b2', self.l1b2)
            inputre0 = tf.slice(self.model.output, [0, 0, 0, 0], [-1, 36, 48, 3])
            inputre0 = inputre0 - tf.reduce_min(inputre0, axis=[1, 2, 3], keep_dims=True)
            inputre0 = inputre0 / tf.reduce_max(inputre0, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('reconstructed0', inputre0, max_outputs=1)
            inputre1 = tf.slice(self.model.output, [0, 0, 0, 3], [-1, 36, 48, 3])
            inputre1 = inputre1 - tf.reduce_min(inputre1, axis=[1, 2, 3], keep_dims=True)
            inputre1 = inputre1 / tf.reduce_max(inputre1, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('reconstructed1', inputre1, max_outputs=1)
            inputre2 = tf.slice(self.model.output, [0, 0, 0, 6], [-1, 36, 48, 3])
            inputre2 = inputre2 - tf.reduce_min(inputre2, axis=[1, 2, 3], keep_dims=True)
            inputre2 = inputre2 / tf.reduce_max(inputre2, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('reconstructed2', inputre2, max_outputs=1)
            inputre3 = tf.slice(self.model.output, [0, 0, 0, 9], [-1, 36, 48, 3])
            inputre3 = inputre3 - tf.reduce_min(inputre3, axis=[1, 2, 3], keep_dims=True)
            inputre3 = inputre3 / tf.reduce_max(inputre3, axis=[1, 2, 3], keep_dims=True)
            self.logger.log_image('reconstructed3', inputre3, max_outputs=1)
        else:
            inputre = self.model.output
            inputre = inputre - tf.reduce_min(inputre, axis=[1, 2], keep_dims=True)
            inputre = inputre / tf.reduce_max(inputre, axis=[1, 2], keep_dims=True)
            self.logger.log_image('reconstructed', inputre)
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
                        latentloss, mse, train_loss, train_summary, _ = session.run(
                            [self.latent_loss, self.losslmse, self.loss, self.logger.summary_op, self.train_op_0],
                            feed_dict={self.handle: train_handle,
                                       self.steps: step2,
                                       self.model.network['keep_prob']: 0.5,
                                       self.model.network['is_training']: 1})

                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            print('{}: {} - Iteration: [{:3}]\t Training_mse_Loss: {:6f}\t Training_Loss: {:6f}\t Latent_Loss: {:6f} '.format(
                                    datetime.now(), FLAGS.exp_name, step, mse, train_loss, latentloss))

                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))

                        # Update counters and stats
                        step += 1
                        step2 += 1
                    except tf.errors.OutOfRangeError:
                        break

                session.run(eval_iterat.initializer)
                # Evaluate model on validation set
                total_loss = self._evaluate(session, 'validation', evaluation_handle, step2)

                print('{}: {} - Epoch: {}\t Validation_mse_Loss: {:6f}'.format(datetime.now(),
                                                                                                        FLAGS.exp_name,
                                                                                                        epoch,
                                                                                                        total_loss))

                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss)
                   # tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)

                self.logger.flush_writer()
                # if accuracy or loss decrease save model
                if total_loss <= best_loss:
                    best_epoch = epoch
                    #best_accuracy = total_accuracy
                    best_loss = total_loss
                    # Save model
                    self._save_checkpoint(session, epoch)
                    with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/model.txt", "w") as outfile:
                        outfile.write(
                            '{}: {}\nBest Epoch: {}\nValidation_mse_Loss: {:6f}\n'.format(
                                datetime.now(),
                                FLAGS.exp_name,
                                best_epoch,
                                best_loss))
            print('{}: {} - Best Epoch: {}\t Validation_mse_Loss: {:6f}'.format(datetime.now(),
                                                                                                         FLAGS.exp_name,
                                                                                                         best_epoch,
                                                                                                         best_loss))

    def _save_checkpoint(self, session, epoch):

        checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
        model_name = 'epoch_{}.ckpt'.format(epoch)
        print('{}: {} - Saving model to {}/{}'.format(datetime.now(), FLAGS.exp_name, checkpoint_dir, model_name))

        self.saver.save(session, '{}/{}'.format(checkpoint_dir, model_name))

    def _valid(self, session, evaluation_handle, steps):
        return self._evaluate(session, 'validation', evaluation_handle, steps)

    def _evaluate(self, session, mod, eval_handle, steps):

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
                one_hot_labels, batch_loss = session.run(
                    [self.labels, self.loss],
                    feed_dict={self.handle: eval_handle,
                               self.steps: steps,
                               self.model.network['keep_prob']: 1.0,
                               self.model.network['is_training']: 0})
                # Update counters
                data_set_size += one_hot_labels.shape[0]
                loss_sum += batch_loss * one_hot_labels.shape[0]
                #accuracy_sum += batch_accuracy * one_hot_labels.shape[0]

            except tf.errors.OutOfRangeError:
                break

        total_loss = loss_sum / data_set_size
        #total_accuracy = accuracy_sum / data_set_size
        #if mod == 'test':
        #   self.plot_confusion_matrix(pred, label)
        return total_loss

    def _retrieve_batch(self, next_batch):

        if FLAGS.model == 'UNet':
            logenergy = tf.slice(next_batch[0], [0, 0, 0, 0, 0], [-1, 1, 36, 48, 1])
            logenergy = tf.reshape(logenergy, shape=[-1, 36, 48, 1])
            mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
            mfcc = tf.image.resize_bilinear(mfcc, [193, 257],
                                                 align_corners=False)
            images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
            acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])
            labels = tf.reshape(next_batch[3], shape=[-1, self.num_classes])
        else:
            raise ValueError('Unknown model type')

        return acoustic, logenergy, mfcc, images, labels

    def test(self, test_data=None):

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
            test_loss = self._evaluate(session, 'test', evaluation_handle, 20000000)

        with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/test_accuracy.txt", "w") as outfile:
            outfile.write('{}: {} - Testing_Loss: {:6f}'.format(datetime.now(),
                                                                                         FLAGS.exp_name,
                                                                                         test_loss))
        print('{}: {} - Testing_Loss: {:6f}'.format(datetime.now(),
                                                                             FLAGS.exp_name,
                                                                             test_loss))

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
