from datetime import datetime
from sklearn.cluster import KMeans
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

    def __init__(self, modelac, modelaudio, modelimages, logger=None, display_freq=1,
                 learning_rate=0.0001, num_classes=14, num_epochs=1, nr_frames=12, temporal_pooling=False):

        self.modelac = modelac
        self.modelaudio = modelaudio
        self.modelimages = modelimages
        self.logger = logger
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling
        self.shape = [self.modelac.height, self.modelac.width, self.modelac.channels]

    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        # self.nproxy = 22
        #
        # self.kmean = tf.placeholder(tf.float32, shape=[self.nproxy, 128])
        # self.proxy = tf.Variable(tf.zeros_like(self.kmean))
        # self.proxy_init = tf.assign(self.proxy, self.kmean)

        # initialize proxies s.t. norm of each proxy ~1 norm
        # self.proxy = tf.Variable(tf.random.normal([self.nproxy, 128], mean=0.0, stddev=0.001, dtype=tf.dtypes.float32))

        # mean = np.mean(batch[0], 0)
        # mean = np.tile(mean, (nproxy, 1))
        # mean = mean + np.random.normal(scale=0.001, size=(nproxy, img_size))
        # np.shape(mean)
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.acoustic, self.mfcc, self.video, self.labels, self.scenario = self._retrieve_batch(next_batch)
        # self.mfcc = self.mfcc - tf.reduce_min(self.mfcc, axis=[1, 2], keep_dims=True)
        # self.mfcc = self.mfcc/tf.reduce_max(self.mfcc, axis=[1, 2], keep_dims=True)

        #UNet
        self.modelac._build_model(self.acoustic)
        self.modelaudio._build_model(self.mfcc)
        self.modelimages._build_model(self.video)

        self.losslmseac = tf.losses.mean_squared_error(self.acoustic, self.modelac.output)
        self.losslmseaudio = tf.losses.mean_squared_error(self.mfcc, self.modelaudio.output)
        self.losslmsevideo = tf.losses.mean_squared_error(self.video, self.modelimages.output)
        self.losslmse = self.losslmseac + self.losslmseaudio + self.losslmsevideo

        self.l1ac = tf.losses.huber_loss(self.acoustic, self.modelac.output)
        self.l1audio = tf.losses.huber_loss(self.mfcc, self.modelaudio.output)
        self.l1video = tf.losses.huber_loss(self.video, self.modelimages.output)
        self.l1 = self.l1ac + self.l1audio + self.l1video

        self.latent_lossac = 0.5 * tf.reduce_sum(tf.square(self.modelac.mean) + tf.square(self.modelac.variance)
                                               - tf.log(1e-8 + tf.square(self.modelac.variance)) - 1, 1)
        self.latent_lossaudio = 0.5 * tf.reduce_sum(tf.square(self.modelaudio.mean) + tf.square(self.modelaudio.variance)
                                                 - tf.log(1e-8 + tf.square(self.modelaudio.variance)) - 1, 1)
        self.latent_lossvideo = 0.5 * tf.reduce_sum(tf.square(self.modelimages.mean) + tf.square(self.modelimages.variance)
                                                 - tf.log(1e-8 + tf.square(self.modelimages.variance)) - 1, 1)
        self.latent_loss = self.latent_lossac + self.latent_lossaudio + self.latent_lossvideo
        # offset = tf.range(12)
        # channels = tf.map_fn(lambda o: tf.slice(considered_modality, [0, 0, 0, o], [-1, 36, 48, 1]),
        #                           offset, dtype=considered_modality.dtype)
        # channelstrue = tf.map_fn(lambda o: tf.slice(self.modelac.output, [0, 0, 0, o], [-1, 36, 48, 1]),
        #                           offset, dtype=self.modelac.output.dtype)
        # self.l2losses = tf.reduce_mean(tf.pow(channels-channelstrue, 2), axis=[1, 2, 3, 4]) / 2
        samples = tf.random_normal([tf.shape(self.modelac.variance)[0], tf.shape(self.modelac.variance)[1]], 0, 1, dtype=tf.float32)
        self.guessed_z_ac = self.modelac.mean + (self.modelac.variance * samples)
        self.guessed_z_audio = self.modelaudio.mean + (self.modelaudio.variance * samples)
        self.guessed_z_video = self.modelimages.mean + (self.modelimages.variance * samples)

        self.latent_loss = tf.reduce_mean(self.latent_loss, 0)/1000000

        # distac = self.d(self.guessed_z_ac, self.proxy)
        # distac = distac - tf.reduce_min(distac, axis=1)
        # distac = distac / tf.reduce_max(distac, axis=1)
        # pyac, idx_py = tf.reduce_min(distac, axis=1), tf.arg_min(distac, dimension=1)
        # sumexp_lessac = tf.reduce_sum(tf.exp(-distac), axis=1)
        # lossac = pyac + tf.log(1e-15 + sumexp_lessac)
        # lossac = tf.reduce_mean(lossac, 0)
        #
        # num = tf.range(tf.shape(idx_py)[0], dtype=tf.int32)
        # num = tf.reshape(num, [-1, 1])
        # idx_py = tf.cast(idx_py, tf.int32)
        # idx_py = tf.reshape(idx_py, [-1, 1])
        # off = tf.concat((num, idx_py), 1)
        #
        # distaudio = self.d(self.guessed_z_audio, self.proxy)
        # distaudio = distaudio - tf.reduce_min(distaudio)
        # distaudio = distaudio / tf.reduce_max(distaudio)
        # cropped = tf.map_fn(lambda i: tf.slice(distaudio, [i[0], i[1]], [1, 1]), off, dtype=tf.float32)
        # pyaudio = tf.reshape(cropped, [-1])
        # #tf.reduce_min(distaudio, axis=1)
        # sumexp_lessaudio = tf.reduce_sum(tf.exp(-distaudio), axis=1)
        # lossaudio = pyaudio + tf.log(1e-15 + sumexp_lessaudio)
        # lossaudio = tf.reduce_mean(lossaudio, 0)
        #
        # distframes = self.d(self.guessed_z_video, self.proxy)
        # distframes = distframes - tf.reduce_min(distframes)
        # distframes = distframes / tf.reduce_max(distframes)
        # cropped = tf.map_fn(lambda i: tf.slice(distframes, [i[0], i[1]], [1, 1]), off, dtype=tf.float32)
        # pyframes = tf.reshape(cropped, [-1])
        # #tf.reduce_min(distframes, axis=1)
        # sumexp_lessframes = tf.reduce_sum(tf.exp(-distframes), axis=1)
        # lossframes = pyframes + tf.log(1e-15 + sumexp_lessframes)
        # lossframes = tf.reduce_mean(lossframes, 0)
        #
        # self.ncaproxyanchor = lossac + lossframes + lossaudio

        lossvideo = self.nca(self.guessed_z_ac, self.guessed_z_video, self.labels, self.scenario)
        lossaudio = self.nca(self.guessed_z_ac, self.guessed_z_audio, self.labels, self.scenario)
        self.ncaproxyanchor = lossvideo + lossaudio

        var_list = self.modelac.train_vars + self.modelaudio.train_vars + self.modelimages.train_vars

        self.loss = self.ncaproxyanchor + self.latent_loss + tf.losses.get_total_loss()
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
                var_list = slim.get_variables(self.modelaudio.scope)
                #var2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.modelaudio.scope)
                # variables_in_checkpoint = tf.train.list_variables()

                saver = tf.train.Saver(var_list=var_list)
                saver.restore(session, FLAGS.audio_init_checkpoint)
                print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            if FLAGS.visual_init_checkpoint is not None:
                var_list = slim.get_variables(self.modelimages.scope + '/')
                # to_exclude = [i.name for i in tf.global_variables()
                #                               if '/Adam' in i.name or 'power' in i.name or 'step' in i.name or
                #                                 'UNetAcoustic' in i.name or 'UNetAudio' in i.name]
                # var_list = slim.get_variables_to_restore(exclude=to_exclude)
                saver = tf.train.Saver(var_list=var_list)
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
        self.logger.log_scalar('ncaproxyanchor_loss', self.ncaproxyanchor)
        # self.logger.log_image('image', self.video, max_outputs=1)
        # self.logger.log_image('sound', self.mfcc, max_outputs=1)
        # # for i in range(12):
        # #     mfcc1 = tf.squeeze(tf.slice(self.l2losses, [i], [1]))
        # #     self.logger.log_scalar('l2_mfcc{}'.format(i), mfcc1)
        # mfccelement0 = tf.slice(self.acoustic, [0, 0, 0, 0], [-1, 36, 48, 3])
        # mfccelement0 = mfccelement0 - tf.reduce_min(mfccelement0, axis=[1, 2, 3], keep_dims=True)
        # mfccelement0 = mfccelement0 / tf.reduce_max(mfccelement0, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('mfcc0', mfccelement0, max_outputs=1)
        # mfccelement1 = tf.slice(self.acoustic, [0, 0, 0, 3], [-1, 36, 48, 3])
        # mfccelement1 = mfccelement1 - tf.reduce_min(mfccelement1, axis=[1, 2, 3], keep_dims=True)
        # mfccelement1 = mfccelement1 / tf.reduce_max(mfccelement1, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('mfcc1', mfccelement1, max_outputs=1)
        # mfccelement2 = tf.slice(self.acoustic, [0, 0, 0, 6], [-1, 36, 48, 3])
        # mfccelement2 = mfccelement2 - tf.reduce_min(mfccelement2, axis=[1, 2, 3], keep_dims=True)
        # mfccelement2 = mfccelement2 / tf.reduce_max(mfccelement2, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('mfcc2', mfccelement2, max_outputs=1)
        # mfccelement3 = tf.slice(self.acoustic, [0, 0, 0, 9], [-1, 36, 48, 3])
        # mfccelement3 = mfccelement3 - tf.reduce_min(mfccelement3, axis=[1, 2, 3], keep_dims=True)
        # mfccelement3 = mfccelement3 / tf.reduce_max(mfccelement3, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('mfcc3', mfccelement3, max_outputs=1)

        # self.logger.log_image('input', self.logenergy, max_outputs=1)

        #UNet
        # inputre0 = tf.slice(self.modelac.output, [0, 0, 0, 0], [-1, 36, 48, 3])
        # inputre0 = inputre0 - tf.reduce_min(inputre0, axis=[1, 2, 3], keep_dims=True)
        # inputre0 = inputre0 / tf.reduce_max(inputre0, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('reconstructed0', inputre0, max_outputs=1)
        # inputre1 = tf.slice(self.modelac.output, [0, 0, 0, 3], [-1, 36, 48, 3])
        # inputre1 = inputre1 - tf.reduce_min(inputre1, axis=[1, 2, 3], keep_dims=True)
        # inputre1 = inputre1 / tf.reduce_max(inputre1, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('reconstructed1', inputre1, max_outputs=1)
        # inputre2 = tf.slice(self.modelac.output, [0, 0, 0, 6], [-1, 36, 48, 3])
        # inputre2 = inputre2 - tf.reduce_min(inputre2, axis=[1, 2, 3], keep_dims=True)
        # inputre2 = inputre2 / tf.reduce_max(inputre2, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('reconstructed2', inputre2, max_outputs=1)
        # inputre3 = tf.slice(self.modelac.output, [0, 0, 0, 9], [-1, 36, 48, 3])
        # inputre3 = inputre3 - tf.reduce_min(inputre3, axis=[1, 2, 3], keep_dims=True)
        # inputre3 = inputre3 / tf.reduce_max(inputre3, axis=[1, 2, 3], keep_dims=True)
        # self.logger.log_image('reconstructed3', inputre3, max_outputs=1)
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
            # session.run(train_iterat.initializer)
            # features_normalized = session.run(self.guessed_z_ac, feed_dict={self.handle: train_handle,
            #                                                                 self.modelac.network['keep_prob']: 0.5,
            #                                                                 self.modelac.network['is_training']: 1})
            # featuresmedia = self.centers(features_normalized)

            # featuresmedia = np.mean(features, axis=0)
            # featuresmedia = np.reshape(featuresmedia, (1,img_size))
            # featuresmedia = np.tile(featuresmedia, (nproxy, 1))
            # session.run(self.proxy_init, feed_dict={self.kmean: featuresmedia})
            # For each epoch
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                session.run(train_iterat.initializer)
                step = 0

                # Initialize iterator over the training set
                # For each mini-batch
                while True:
                    try:
                        # Forward batch through the network
                        latentloss, mse, train_loss, train_summary, ncaproxyanchor, _ = session.run([self.latent_loss, self.losslmse,
                            self.loss, self.logger.summary_op, self.ncaproxyanchor, self.train_op_0],  # self.train_op_1
                                    feed_dict={self.handle: train_handle,
                                       self.modelac.network['keep_prob']: 0.5,
                                       self.modelac.network['is_training']: 1,
                                       self.modelaudio.network['keep_prob']: 0.5,
                                       self.modelaudio.network['is_training']: 1,
                                       self.modelimages.network['keep_prob']: 0.5,
                                       self.modelimages.network['is_training']: 1})

                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            print('{}: {} - Iteration: [{:3}]\t Training_mse_Loss: {:6f}\t NCA_Loss:'
                                  ' {:6f}\t Latent_Loss: {:6f} '.format(
                                    datetime.now(), FLAGS.exp_name, step, mse, ncaproxyanchor, latentloss))

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
                    [self.labels, self.loss],
                    feed_dict={self.handle: eval_handle,
                               self.modelac.network['keep_prob']: 1.0,
                               self.modelac.network['is_training']: 0,
                               self.modelaudio.network['keep_prob']: 1.0,
                               self.modelaudio.network['is_training']: 0,
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
            mfcc = tf.reshape(next_batch[1], shape=[-1, 99, 257, 1])
            images = tf.reshape(next_batch[2], shape=[-1, 224, 298, 3])
            acoustic = tf.reshape(next_batch[0], shape=[-1, 36, 48, 12])
            if FLAGS.datatype == 'music':
                num_actions = 9
                num_locations = 11  # maximum number of videos for a class
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
            test_loss = self._evaluate(session, 'test', evaluation_handle)

        with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/test_accuracy.txt", "w") as outfile:
            outfile.write('{}: {} - Testing_Loss: {:6f}'.format(datetime.now(),
                                                                                         FLAGS.exp_name,
                                                                                         test_loss))
        print('{}: {} - Testing_Loss: {:6f}'.format(datetime.now(),
                                                                             FLAGS.exp_name,
                                                                             test_loss))

        return test_loss

    def centers(self, features):
        kmeans = KMeans(n_clusters=self.nproxy, n_init=4, n_jobs=-1)
        kmeans.fit(features)
        centers = kmeans.cluster_centers_
        return centers

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

    def d(self, X, Y):
      """
      Computes pairwise distance between each pair of points in two sets
      Args:
        X - [N,D] matrix representing N D-dimensional vectors
        Y - [M,D] matrix representing M D-dimensional vectors
      Returns:
        [N,M] matrix of (squared) Euclidean distances
      """
      return (tf.reduce_sum(X * X, 1, True) -
              2 * tf.matmul(X, tf.transpose(Y)) +
              tf.transpose(tf.reduce_sum(Y * Y, 1, True)))

    def nca(self, data0, data1, labels, scenario):  # acoustic_data and video_data
        # compute distances
        # pairwise_dist = tf.reduce_sum(tf.square(data0 - data1), -1)
        pairwise_dist = self._pairwise_distances(data0, data1)
        pairwise_dist = pairwise_dist - tf.reduce_min(pairwise_dist, 1)
        pairwise_dist = pairwise_dist / tf.reduce_max(pairwise_dist, 1)
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
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

        # shape (batch_size,)
        sum_negative_dist = tf.reduce_sum(tf.exp(-anchor_negative_dist), axis=1, keep_dims=True)
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        nca_loss = hardest_positive_dist + tf.log(1e-15 + sum_negative_dist)
        nca_loss = tf.reduce_mean(nca_loss)

        return nca_loss

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

