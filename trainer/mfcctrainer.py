from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from logger.logger import Logger
flags = tf.app.flags
FLAGS = flags.FLAGS
_FRAMES_PER_SECOND = 12


class Trainer(object):

    def __init__(self, modelac, modelimages, display_freq=1,
                 learning_rate=0.0001, num_classes=14, num_epochs=1, nr_frames=12, temporal_pooling=False):

        self.modelac = modelac
        self.modelimages = modelimages
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling

    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.acoustic, self.mfcc, self.video, self.labels, self.scenario = self._retrieve_batch(next_batch)
        # self.mfcc = self.mfcc - tf.reduce_min(self.mfcc, axis=[1], keep_dims=True)
        # self.mfcc = self.mfcc/tf.reduce_max(self.mfcc, axis=[1], keep_dims=True)
        self.mfccmap = tf.reshape(self.mfcc, (-1, 1, 12))
        self.mfccmap = tf.tile(self.mfccmap, (1, 36*48, 1))
        self.mfccmap = tf.reshape(self.mfccmap, (-1, 36, 48, 12))
        #build video model
        self.modelimages._build_model(self.video)

        #build acoustic images with mean and std
        self.modelac._build_model(self.mfccmap, self.modelimages.output)
        if FLAGS.MSE:
            self.lossmse = tf.losses.mean_squared_error(self.acoustic, self.modelac.output)

        if FLAGS.huber_loss:
            self.l1 = tf.losses.huber_loss(self.acoustic, self.modelac.output)
            if not FLAGS.MSE:
                self.lossmse = self.l1

        if not FLAGS.ae:
            #latent loss on associator
            self.latent_lossac = 0.5 * tf.reduce_sum(tf.square(self.modelac.mean) + tf.square(self.modelac.std)
                                                   - tf.log(1e-8 + tf.square(self.modelac.std)) - 1, 1)
            self.latent_loss = self.latent_lossac
            self.latent_loss = FLAGS.latent_loss * tf.reduce_mean(self.latent_loss, 0)
            self.loss = self.latent_loss + tf.losses.get_total_loss()
        else:
            self.loss = tf.losses.get_total_loss()

        var_list = self.modelac.train_vars + self.modelimages.train_vars
        # slim.get_model_variables(self.scope + '/logits')
        # Initialize counters and stats
        self.global_step = tf.train.create_global_step()

        # Define optimizer
        # before different
        # self.optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0000001)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.device('/gpu:0'):
                # Compute the gradients for embedding
                self.train_op_0 = self.optimizer.minimize(loss=self.loss,
                                                           var_list=var_list,
                                                           global_step=self.global_step)
        # Initialize model saver
        self.saver = tf.train.Saver(max_to_keep=11)#each 10 epochs and best
        return iterat

    def _build_functionstest(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.acoustic, self.mfcc, self.video, self.labels, self.scenario = self._retrieve_batch(next_batch)
        # self.mfcc = self.mfcc - tf.reduce_min(self.mfcc, axis=[1], keep_dims=True)
        # self.mfcc = self.mfcc/tf.reduce_max(self.mfcc, axis=[1], keep_dims=True)
        self.mfccmap = tf.reshape(self.mfcc, (-1, 1, 12))
        self.mfccmap = tf.tile(self.mfccmap, (1, 36*48, 1))
        self.mfccmap = tf.reshape(self.mfccmap, (-1, 36, 48, 12))
        #build video model
        self.modelimages._build_model(self.video)

        #build acoustic images with mean and std
        self.modelac._build_model(self.mfccmap, self.modelimages.output)
        # if FLAGS.MSE:
        self.lossmse = tf.losses.mean_squared_error(self.acoustic, self.modelac.output)

        mfccelement0 = tf.slice(self.acoustic, [0, 0, 0, 0], [-1, 36, 48, 3])
        mfccelement1 = tf.slice(self.acoustic, [0, 0, 0, 3], [-1, 36, 48, 3])
        mfccelement2 = tf.slice(self.acoustic, [0, 0, 0, 6], [-1, 36, 48, 3])
        mfccelement3 = tf.slice(self.acoustic, [0, 0, 0, 9], [-1, 36, 48, 3])

        inputre0 = tf.slice(self.modelac.output, [0, 0, 0, 0], [-1, 36, 48, 3])
        inputre1 = tf.slice(self.modelac.output, [0, 0, 0, 3], [-1, 36, 48, 3])
        inputre2 = tf.slice(self.modelac.output, [0, 0, 0, 6], [-1, 36, 48, 3])
        inputre3 = tf.slice(self.modelac.output, [0, 0, 0, 9], [-1, 36, 48, 3])
        self.lossmse0 = tf.losses.mean_squared_error(mfccelement0, inputre0)
        self.lossmse1 = tf.losses.mean_squared_error(mfccelement1, inputre1)
        self.lossmse2 = tf.losses.mean_squared_error(mfccelement2, inputre2)
        self.lossmse3 = tf.losses.mean_squared_error(mfccelement3, inputre3)

        # if FLAGS.huber_loss:
        #     self.l1 = tf.losses.huber_loss(self.acoustic, self.modelac.output)
        #     if not FLAGS.MSE:
        #         self.lossmse = self.l1

        # if not FLAGS.ae:
        #     #latent loss on associator
        #     self.latent_lossac = 0.5 * tf.reduce_sum(tf.square(self.modelac.mean) + tf.square(self.modelac.std)
        #                                            - tf.log(1e-8 + tf.square(self.modelac.std)) - 1, 1)
        #     self.latent_loss = self.latent_lossac
        #     self.latent_loss = FLAGS.latent_loss * tf.reduce_mean(self.latent_loss, 0)
        #     self.loss = self.latent_loss + tf.losses.get_total_loss()
        # else:
        #     self.loss = tf.losses.get_total_loss()

        # var_list = self.modelac.train_vars + self.modelimages.train_vars
        # slim.get_model_variables(self.scope + '/logits')
        # Initialize counters and stats
        self.global_step = tf.train.create_global_step()

        # Define optimizer
        # before different
        # self.optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0000001)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     with tf.device('/gpu:0'):
        #         # Compute the gradients for embedding
        #         self.train_op_0 = self.optimizer.minimize(loss=self.loss,
        #                                                    var_list=var_list,
        #                                                    global_step=self.global_step)
        # Initialize model saver
        # self.saver = tf.train.Saver(max_to_keep=11)#each 10 epochs and best
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
            self.modelac.init_model(session, FLAGS.init_checkpoint)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
        elif FLAGS.acoustic_init_checkpoint is not None or FLAGS.audio_init_checkpoint is not None or \
                FLAGS.visual_init_checkpoint is not None:
            session.run(tf.global_variables_initializer())
            if FLAGS.acoustic_init_checkpoint is not None:
                var_list = slim.get_variables(self.modelac.scope)
                # variables_in_checkpoint = tf.train.list_variables()

                saver = tf.train.Saver(var_list=var_list)
                saver.restore(session, FLAGS.acoustic_init_checkpoint)
                print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            if FLAGS.audio_init_checkpoint is not None:
                var_list = slim.get_variables(self.modelassociator.scope)
                #var2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.modelaudio.scope)
                # variables_in_checkpoint = tf.train.list_variables()
                vars_dict = {}
                for var_current in var_list:
                    # print(var_current.op.name)  # this gets only name
                    v = str.join('/', var_current.op.name.split('/')[1:])
                    v = 'UNetAudio/' + v
                    if 'mean' in var_current.op.name.split('/')[1] or 'std' in var_current.op.name.split('/')[1] or 'layer5' in var_current.op.name.split('/')[1]:
                        print(v)
                        continue
                    # print(v)
                    vars_dict[v] = var_current

                saver = tf.train.Saver(var_list=vars_dict)
                saver.restore(session, FLAGS.audio_init_checkpoint)
                print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            if FLAGS.visual_init_checkpoint is not None:
                var_list = slim.get_variables(self.modelimages.scope + '/')
                # to_exclude = [i.name for i in tf.global_variables()
                #                               if 'logits' in i.name or 'conv_map' in i.name]
                # var_list = slim.get_variables_to_restore(exclude=to_exclude)
                var_list_filter = []
                for var_current in var_list:
                    if 'logits' not in var_current.op.name.split('/')[1] and 'conv_map' not in var_current.op.name.split('/')[1]:
                        var_list_filter.append(var_current)
                saver = tf.train.Saver(var_list=var_list_filter)
                saver.restore(session, FLAGS.visual_init_checkpoint)
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
        session.run(tf.global_variables_initializer())
        var_list1 = slim.get_variables(self.modelimages.scope + '/')

        var_list2 = slim.get_variables(self.modelac.scope + '/')
        # else:
        #     var_list = slim.get_model_variables(self.model.scope)
        saver = tf.train.Saver(var_list=var_list1 + var_list2)
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
        if FLAGS.MSE:
            self.logger.log_scalar('l2 loss', self.lossmse)
        if FLAGS.huber_loss:
            self.logger.log_scalar('l1 loss', self.l1)
        self.logger.log_scalar('train_loss', self.loss)
        if not FLAGS.ae:
            self.logger.log_scalar('latent_loss', self.latent_loss)
        # self.logger.log_image('image', self.video, max_outputs=1)
        # self.logger.log_image('sound', self.mfcc, max_outputs=1)
        # for i in range(12):
        #     mfcc1 = tf.squeeze(tf.slice(self.l2losses, [i], [1]))
        #     self.logger.log_scalar('l2_mfcc{}'.format(i), mfcc1)
        mfccelement0 = tf.slice(self.acoustic, [0, 0, 0, 0], [-1, 36, 48, 3])
        self.logger.log_image('mfcc0', mfccelement0, max_outputs=1)
        mfccelement1 = tf.slice(self.acoustic, [0, 0, 0, 3], [-1, 36, 48, 3])
        self.logger.log_image('mfcc1', mfccelement1, max_outputs=1)
        mfccelement2 = tf.slice(self.acoustic, [0, 0, 0, 6], [-1, 36, 48, 3])
        self.logger.log_image('mfcc2', mfccelement2, max_outputs=1)
        mfccelement3 = tf.slice(self.acoustic, [0, 0, 0, 9], [-1, 36, 48, 3])
        self.logger.log_image('mfcc3', mfccelement3, max_outputs=1)

            # self.logger.log_image('input', self.logenergy, max_outputs=1)

        #UNet
        inputre0 = tf.slice(self.modelac.output, [0, 0, 0, 0], [-1, 36, 48, 3])
        self.logger.log_image('reconstructed0', inputre0, max_outputs=1)
        inputre1 = tf.slice(self.modelac.output, [0, 0, 0, 3], [-1, 36, 48, 3])
        self.logger.log_image('reconstructed1', inputre1, max_outputs=1)
        inputre2 = tf.slice(self.modelac.output, [0, 0, 0, 6], [-1, 36, 48, 3])
        self.logger.log_image('reconstructed2', inputre2, max_outputs=1)
        inputre3 = tf.slice(self.modelac.output, [0, 0, 0, 9], [-1, 36, 48, 3])
        self.logger.log_image('reconstructed3', inputre3, max_outputs=1)
        # inputaudio = self.modelaudio.output
        # inputaudio = inputaudio - tf.reduce_min(inputaudio, axis=[1, 2], keep_dims=True)
        # inputaudio = inputaudio / tf.reduce_max(inputaudio, axis=[1, 2], keep_dims=True)
        # self.logger.log_image('reconstructed audio', inputaudio, max_outputs=1)
        # inputframes = self.modelimages.output
        # inputframes = inputframes - tf.reduce_min(inputframes, axis=[1, 2], keep_dims=True)
        # inputframes = inputframes / tf.reduce_max(inputframes, axis=[1, 2], keep_dims=True)
        # self.logger.log_image('reconstructed rgb', inputframes, max_outputs=1)
        # Add the accuracy to the summary
        #self.logger.log_scalar('train_accuracy', self.accuracy)

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
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                step = 0
                # Initialize iterator over the training set
                session.run(train_iterat.initializer)
                # For each mini-batch
                while True:
                    try:
                        # ac, mf = session.run([self.modelac.network['features'], self.modelimages.output],
                        #             feed_dict={self.handle: train_handle,
                        #                self.modelac.network['keep_prob']: 0.5,
                        #                self.modelac.network['is_training']: 1,
                        #                self.modelimages.network['keep_prob']: 0.5,
                        #                self.modelimages.network['is_training']: 1})
                        # Forward batch through the network
                        mse, train_loss, train_summary, _ = session.run([self.lossmse,
                            self.loss, self.logger.summary_op, self.train_op_0],  # self.train_op_1
                                    feed_dict={self.handle: train_handle,
                                       self.modelac.network['keep_prob']: 0.5,
                                       self.modelac.network['is_training']: 1,
                                       self.modelimages.network['keep_prob']: 0.5,
                                       self.modelimages.network['is_training']: 1})

                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            print('{}: {} - Iteration: [{:3}]\t Training_mse_Loss: {:6f}\t Training_Loss:'
                                  ' {:6f}'.format(
                                    datetime.now(), FLAGS.exp_name, step, mse, train_loss))

                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))

                        # Update counters and stats
                        step += 1

                    except tf.errors.OutOfRangeError:
                        break

                session.run(eval_iterat.initializer)
                # Evaluate model on validation set
                total_loss = self._evaluate(session, 'validation', evaluation_handle)

                print('{}: {} - Epoch: {}\t Validation_mse_Loss: {:6f}'.format(datetime.now(), FLAGS.exp_name,
                                                                                                        epoch,
                                                                                                        total_loss))

                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss)
                   # tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)

                self.logger.flush_writer()
                if epoch % 10 == 0:
                    # Save model
                    self._save_checkpoint(session, epoch)
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
            print('{}: {} - Best Epoch: {}\t Validation_mse_Loss: {:6f}'.format(datetime.now(), FLAGS.exp_name,
                                                                                                         best_epoch,
                                                                                                         best_loss))

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
                one_hot_labels, batch_loss = session.run(
                    [self.labels, self.lossmse],
                    feed_dict={self.handle: eval_handle,
                               self.modelac.network['keep_prob']: 1.0,
                               self.modelac.network['is_training']: 0,
                               self.modelimages.network['keep_prob']: 1.0,
                               self.modelimages.network['is_training']: 0})
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
            mfcc = tf.reshape(next_batch[1], shape=[-1, 12])
            # mfcc = tf.image.resize_bilinear(mfcc, [193, 257], align_corners=False)
            images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
            acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])
            if FLAGS.datatype == 'music':
                num_actions = 9
                num_locations = 11  # maximum number of videos for a class
            elif FLAGS.datatype == 'old':
                num_actions = 14
                num_locations = 3
            else:  # self.datakind == 'outdoor':
                num_actions = 10
                num_locations = 61
            labels = tf.reshape(next_batch[3], shape=[-1, num_actions])
            labels = tf.argmax(labels, axis=1)
            scenario = tf.reshape(next_batch[4], shape=[-1, num_locations])
            scenario = tf.argmax(scenario, axis=1)
        else:
            raise ValueError('Unknown model type')

        return acoustic, mfcc, images, labels, scenario

    def modDrop(self, mean, std, is_training, p_mod=.5):
        on = tf.cast(tf.random_uniform([1]) - p_mod < 0, tf.float32)
        mean = tf.cond(is_training, lambda: on * mean, lambda: mean)
        std = tf.cond(is_training, lambda: on * std, lambda: std)
        return mean, std, on
    #l = self.modDrop(l, self.modelenergy.network['is_training'])

    def test(self, test_data=None):

        # Assert testing set is not None
        assert test_data is not None
        eval_iterat = self._build_functionstest(test_data)

        # Start training session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint if given
            self._restore_model(session)
            session.run(eval_iterat.initializer)
            # Evaluate model over the testing set
            # Initialize counters and stats
            loss_sum = 0
            loss_sum0 = 0
            loss_sum1 = 0
            loss_sum2 = 0
            loss_sum3 = 0

            data_set_size = 0
            # For each mini-batch
            name_folder = str.join('/', FLAGS.restore_checkpoint.split('/')[:-1])
            while True:
                try:
                    # Compute batch loss and accuracy
                    one_hot_labels, batch_loss, batch_loss0, batch_loss1, batch_loss2, batch_loss3 = session.run(
                        [self.labels, self.lossmse, self.lossmse0, self.lossmse1, self.lossmse2, self.lossmse3],
                        feed_dict={self.handle: evaluation_handle,
                                   self.modelac.network['keep_prob']: 1.0,
                                   self.modelac.network['is_training']: 0,
                                   self.modelimages.network['keep_prob']: 1.0,
                                   self.modelimages.network['is_training']: 0})
                    # Update counters
                    data_set_size += one_hot_labels.shape[0]
                    loss_sum += batch_loss * one_hot_labels.shape[0]

                    loss_sum0 += batch_loss0 * one_hot_labels.shape[0]
                    loss_sum1 += batch_loss1 * one_hot_labels.shape[0]
                    loss_sum2 += batch_loss2 * one_hot_labels.shape[0]
                    loss_sum3 += batch_loss3 * one_hot_labels.shape[0]

                except tf.errors.OutOfRangeError:
                    break

            test_loss = loss_sum / data_set_size
            test_loss0 = loss_sum0 / data_set_size
            test_loss1 = loss_sum1 / data_set_size
            test_loss2 = loss_sum2 / data_set_size
            test_loss3 = loss_sum3 / data_set_size

        with open('{}'.format(name_folder) + "/test_accuracy_{}.txt".format(
                FLAGS.restore_checkpoint.split('/')[-1].split('.')[0].split('_')[1]), "w") as outfile:
            outfile.write('{} - Testing_Loss: {:6f}\t  Testing_Loss0: {:6f}\t Testing_Loss1:'
                          ' {:6f}\t Testing_Loss2: {:6f}\t Testing_Loss3: {:6f}'.format(datetime.now(), test_loss,
                                                                test_loss0, test_loss1, test_loss2, test_loss3))
        print('{} - Testing_Loss: {:6f}\t  Testing_Loss0: {:6f}\t Testing_Loss1:'
              ' {:6f}\t Testing_Loss2: {:6f}\t Testing_Loss3: {:6f}'.format(datetime.now(), test_loss,
                                                                test_loss0, test_loss1, test_loss2, test_loss3))

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

    def _pairwise_distances(self, embeddings0, embeddings1, squared=True):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product0 = tf.matmul(embeddings0, tf.transpose(embeddings0))
        dot_product1 = tf.matmul(embeddings1, tf.transpose(embeddings1))
        dot_productab = tf.matmul(embeddings0, tf.transpose(embeddings1))
        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm0 = tf.diag_part(dot_product0)
        square_norm1 = tf.diag_part(dot_product1)
        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm0, 0) - 2.0 * dot_productab + tf.expand_dims(square_norm1, 1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances

    def _get_anchor_positive_and_negative_triplet_mask(self, labels, scenario):
        """Return a 2D mask where mask[a, p] is True iff a and p have same label and scenario.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Also if i and j are not distinct is ok because we are considering audio and video embeddings
        # indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        # indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j] and scenario are equal
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        scenario_equal = tf.equal(tf.expand_dims(scenario, 0), tf.expand_dims(scenario, 1))

        # Combine the two masks
        mask = tf.logical_and(scenario_equal, labels_equal)

        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels or scenario.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k] or different scenario
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        maskneg0 = tf.logical_not(labels_equal)
        maskneg1 = tf.logical_not(scenario_equal)
        maskneg = tf.logical_or(maskneg0, maskneg1)
        return mask, maskneg

    def _get_triplet_mask(self, labels, scenario):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - video[i] == video[j] and  video[i] != video[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check if video[i] == video[j] and video[i] != video[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        scenario_equal = tf.equal(tf.expand_dims(scenario, 0), tf.expand_dims(scenario, 1))
        # Combine the three masks
        same_video = tf.logical_and(scenario_equal, label_equal)
        i_equal_j = tf.expand_dims(same_video, 2)
        i_equal_k = tf.expand_dims(same_video, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        return valid_labels

        # compute closest embedding  negative and furthest positive for each batch
        # negative has different label, or person or location
        # positive has same label, person and location

    def mix_data_hard(self, data0, data1, labels, scenario, margin):  # acoustic_data and video_data
        # compute distances
        # pairwise_dist = tf.reduce_sum(tf.square(data0 - data1), -1)
        pairwise_dist = self._pairwise_distances(data0, data1)
        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label, person and location)
        mask_anchor_positive, mask_anchor_negative = self._get_anchor_positive_and_negative_triplet_mask(labels,
                                                                                                         scenario)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)
        # We put to 0 any element where (a, p) is not valid (valid if is the same video)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keep_dims=True)
        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels, or location, or person)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n) or location, or person)
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keep_dims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keep_dims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        mask = self._get_triplet_mask(labels, scenario)
        mask = tf.to_float(mask)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss, fraction_positive_triplets

    def mix_all(self, data0, data1, labels, scenario, margin):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(data0, data1)
        # compute distances
        # pairwise_dist = tf.reduce_sum(tf.square(data0 - data1), -1)

        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where video(a) != video(p) or video(n) == video(a))
        mask = self._get_triplet_mask(labels, scenario)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets
