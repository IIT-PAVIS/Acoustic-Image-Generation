import tensorflow as tf
from datetime import datetime
from logger.logger import Logger
from models.multimodal import JointTwomvae
from models.multimodal import AssociatorVideoAc
from models.multimodal import AssociatorAudio
from models.multimodal import AssociatorAudioAc
from models.multimodal import Jointmvae
from models.multimodal import JointTwomvae2
from models.unet_sound22 import UNetSound as UNetSound22
from models.unet_sound2 import UNetSound
from models.unet_architecture_energy import UNetE
from models.unet_noconc2 import UNetAc as UNetAc2
from models.unet_z import UNetAc as UNetzvariable
from models.unet_noconc import UNetAc
from models.unet_acresnet2skip import UNetAc as UNetAcResNet50_2skips
from models.unet_acresnet import UNetAc as UNetAcResNet50
from models.unet_acresnet0skip import UNetAc as UNetAcResNet50_0skips
from models.unet_architecture_noconc import UNet
from models.unet_architecture_noconc2 import UNet as Unet2
from models.vision import ResNet50Model
from trainer.trainermulti import Trainer as TrainerMulti
from trainer.trainer import Trainer as Trainer
from trainer.trainer_three import Trainer as TrainerLoss
from trainer.trainer2 import Trainer as TrainerNCAproxyanchor
from trainer.mfcctrainer import Trainer as TrainerMask
from trainer.trainer_proietta import Trainer as TrainerProject
from dataloader.actions_data_old import ActionsDataLoader
from dataloader.outdoor_data_mfcc import ActionsDataLoader as SoundDataLoader
from trainer.trainer_class import Trainer as Trainer_classification
from trainer.trainer_reconstructed_class import Trainer as Trainer_rec_class
from models.dualcamnet import DualCamHybridModel
flags = tf.app.flags
flags.DEFINE_string('mode', None, 'Execution mode, it can be either \'train\' or \'test\'')
flags.DEFINE_string('model', None, 'Model type, it can be one of \'SeeNet\', \'ResNet50\', \'TemporalResNet50\', '
                                   '\'DualCamNet\', \'DualCamHybridNet\', \'SoundNet5\', or \'HearNet\'')
flags.DEFINE_string('train_file', None, 'Path to the plain text file for the training set')
flags.DEFINE_string('valid_file', None, 'Path to the plain text file for the validation set')
flags.DEFINE_string('test_file', None, 'Path to the plain text file for the testing set')
flags.DEFINE_string('exp_name', None, 'Name of the experiment')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_string('acoustic_init_checkpoint', None, 'Checkpoint file for acoustic model initialization')
flags.DEFINE_string('audio_init_checkpoint', None, 'Checkpoint file for audio model initialization')
flags.DEFINE_string('visual_init_checkpoint', None, 'Checkpoint file for visual model initialization')
flags.DEFINE_string('restore_checkpoint', None, 'Checkpoint file for session restoring')
flags.DEFINE_integer('batch_size', 8, 'Size of the mini-batch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('latent_loss', 0.000001, 'Learning rate')
flags.DEFINE_integer('display_freq', 1, 'How often must be shown training results')
flags.DEFINE_integer('num_epochs', 100, 'Number of iterations through dataset')
flags.DEFINE_integer('total_length', 30, 'Length in seconds of a full sequence')
# sample length is 1 s for dualcamnet and 5 s for hearnet and soundnet
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
# number of crops 30 for 1 s and 6 for 5 s
flags.DEFINE_integer('number_of_crops', 30, 'Number of crops')
flags.DEFINE_integer('buffer_size', 100, 'Size of pre-fetch buffer')
flags.DEFINE_string('tensorboard', None, 'Directory for storing logs')
flags.DEFINE_string('checkpoint_dir', None, 'Directory for storing models')
flags.DEFINE_integer('temporal_pooling', 0, 'Flag to indicate whether to use average pooling over time')
flags.DEFINE_integer('embedding', 0, 'Say if you are training 128 vectors')
flags.DEFINE_float('margin', 0.2, 'margin')  # between 0 and 11 for 128 vector
flags.DEFINE_integer('block_size', 1, 'Number of frames to pick randomly for each second')  # 12
flags.DEFINE_integer('num_class', 128, 'Classes')
flags.DEFINE_string('datatype', 'outdoor', 'music or outdoor or old')
flags.DEFINE_integer('correspondence', 0, 'use correspondence')
flags.DEFINE_integer('proxy', 0, 'Use NCA')
flags.DEFINE_string('encoder_type', 'Video', 'Modality for encoder, it can be one of \'Energy\', \'Video\', \'Ac\' or \'Audio\'')
flags.DEFINE_integer('fusion', 0, 'Use both audio and video')
flags.DEFINE_integer('moddrop', 0, 'Use audio video and dropmod ac')
flags.DEFINE_integer('l2', 0, 'Use l2 between latent variables')
flags.DEFINE_integer('project', 0, 'Use conversion between latent variables')
flags.DEFINE_integer('jointmvae', 0, 'Use joint latent')
flags.DEFINE_integer('onlyaudiovideo', 0, 'Using only audio and video')
flags.DEFINE_integer('mfcc', 0, 'Using mfcc and resnet50 or dualcamnet with acoustic and mfccmap')
flags.DEFINE_integer('mfccmap', 0, 'Do not reconstruct')
flags.DEFINE_integer('num_skip_conn', 1, 'Number of skip')
flags.DEFINE_integer('ae', 0, 'auto encoder')
flags.DEFINE_integer('MSE', 1, 'MSE loss use')
flags.DEFINE_integer('huber_loss', 1, 'Huber loss use')
FLAGS = flags.FLAGS


def main(_):

    # Create data loaders according to the received program arguments
    print('{}: {} - Creating data loaders'.format(datetime.now(), FLAGS.exp_name))

    # random_pick = (FLAGS.model == 'TemporalResNet50' or FLAGS.model_1 == 'TemporalResNet50') or (FLAGS.model == 'ResNet18' or FLAGS.model_1 == 'ResNet18')
    # if we are randomly picking total number of frames, we can set random pick to False
    nr_frames = FLAGS.block_size * FLAGS.sample_length
    # if (FLAGS.model == 'ResNet18_v1' or FLAGS.model == 'ResNet50' or FLAGS.model_1 == 'ResNet18_v1'
    #     or FLAGS.model_1 == 'ResNet50' or FLAGS.model == 'AVNet') and nr_frames < 12*FLAGS.sample_length:
    #     random_pick = True
    # else:
    #     random_pick = False
    random_pick = False
    build_spectrogram = True
    normalize = False #normalize spectrogram without statistic every one

    modalities = []
    #consider all
    modalities.append(0)
    modalities.append(1)
    modalities.append(2)

    with tf.device('/cpu:0'):
        if FLAGS.datatype == 'old':
            num_classes = 14
            if FLAGS.train_file is None:
                train_data = None
            else:
                train_data = ActionsDataLoader(FLAGS.train_file, 'training', FLAGS.batch_size, num_epochs=1,
                                        sample_length=FLAGS.sample_length, embedding=FLAGS.embedding,
                                        buffer_size=FLAGS.buffer_size, datakind=FLAGS.datatype,
                                        shuffle=True, normalize=normalize, random_pick=random_pick,
                                        correspondence=FLAGS.correspondence,
                                        build_spectrogram=build_spectrogram, modalities=modalities, nr_frames=nr_frames)

            if FLAGS.valid_file is None:
                valid_data = None
            else:
                valid_data = ActionsDataLoader(FLAGS.valid_file, 'validation', FLAGS.batch_size, num_epochs=1,
                                        sample_length=FLAGS.sample_length, datakind=FLAGS.datatype, embedding=FLAGS.embedding,
                                        buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                        correspondence=FLAGS.correspondence,
                                        random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities,
                                        nr_frames=nr_frames)

            if FLAGS.test_file is None:
                test_data = None
            else:
                test_data = ActionsDataLoader(FLAGS.test_file, 'testing', FLAGS.batch_size, num_epochs=1,
                                       sample_length=FLAGS.sample_length, datakind=FLAGS.datatype, embedding=FLAGS.embedding,
                                       buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                       correspondence=FLAGS.correspondence,
                                       random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities,
                                       nr_frames=nr_frames)
        elif FLAGS.datatype == 'outdoor':
            num_classes = 10
            if FLAGS.train_file is None:
                train_data = None
            else:
                train_data = SoundDataLoader(FLAGS.train_file, 'training', FLAGS.batch_size, num_epochs=1,
                                        sample_length=FLAGS.sample_length, embedding=FLAGS.embedding,
                                        buffer_size=FLAGS.buffer_size, datakind=FLAGS.datatype,
                                        shuffle=True, normalize=normalize, random_pick=random_pick,
                                        correspondence=FLAGS.correspondence,
                                        build_spectrogram=build_spectrogram, modalities=modalities, nr_frames=nr_frames)

            if FLAGS.valid_file is None:
                valid_data = None
            else:
                valid_data = SoundDataLoader(FLAGS.valid_file, 'validation', FLAGS.batch_size, num_epochs=1,
                                        sample_length=FLAGS.sample_length, datakind=FLAGS.datatype, embedding=FLAGS.embedding,
                                        buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                        correspondence=FLAGS.correspondence,
                                        random_pick=random_pick, build_spectrogram=build_spectrogram,
                                        modalities=modalities,
                                        nr_frames=nr_frames)

            if FLAGS.test_file is None:
                test_data = None
            else:
                test_data = SoundDataLoader(FLAGS.test_file, 'testing', FLAGS.batch_size, num_epochs=1,
                                       sample_length=FLAGS.sample_length, datakind=FLAGS.datatype, embedding=FLAGS.embedding,
                                       buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                       correspondence=FLAGS.correspondence,
                                       random_pick=random_pick, build_spectrogram=build_spectrogram,
                                       modalities=modalities,
                                       nr_frames=nr_frames)

    # Build model
    print('{}: {} - Building model'.format(datetime.now(), FLAGS.exp_name))

    if FLAGS.embedding:
        with tf.device('/gpu:0'):
            if FLAGS.project:
                model_encoder_images = UNet(input_shape=[224, 298, 3])
                model_encoder_audio = UNetSound(input_shape=[193, 257, 1])
                model_encoder_acoustic = UNetzvariable(input_shape=[36, 48, 12])
                if FLAGS.fusion:
                    model_associator = AssociatorVideoAc(input_shape=1024)
                    model_associator1 = AssociatorAudioAc(input_shape=256)
                elif FLAGS.encoder_type == 'Video':
                    model_associator = AssociatorVideoAc(input_shape=1024)
                    model_associator1 = None
                else:
                    model_associator = AssociatorAudio(input_shape=[193, 257, 1])
                    model_associator1 = None
            elif FLAGS.jointmvae:
                model_encoder_images = Unet2(input_shape=[224, 298, 3])
                model_encoder_audio = UNetSound22(input_shape=[193, 257, 1])
                model_encoder_acoustic = UNetAc2(input_shape=[36, 48, 12])
                if FLAGS.fusion:
                    model_associator = JointTwomvae2()
                    model_associator1 = None
                elif FLAGS.onlyaudiovideo:
                    model_associator = Jointmvae()
                    model_associator1 = JointTwomvae()
                else:
                    model_associator = Jointmvae()
                    model_associator1 = None
            else:#mfcc
                model_encoder_images = ResNet50Model(input_shape=[224, 298, 3], num_classes=None)
                if FLAGS.num_skip_conn == 2:
                    model_encoder_acoustic = UNetAcResNet50_2skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)
                elif FLAGS.num_skip_conn == 1:
                    model_encoder_acoustic = UNetAcResNet50(input_shape=[36, 48, 12], embedding=FLAGS.ae)
                elif FLAGS.num_skip_conn == 0:
                    model_encoder_acoustic = UNetAcResNet50_0skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)
            # Build trainer
        print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))

        if FLAGS.proxy == 0:
            if FLAGS.project:
                trainer = TrainerProject(model_encoder_acoustic, model_encoder_audio, model_encoder_images,
                                         model_associator, model_associator1, display_freq=FLAGS.display_freq,
                                         learning_rate=FLAGS.learning_rate, num_classes=num_classes,
                                         num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
            elif FLAGS.jointmvae:
                 trainer = TrainerMulti(model_encoder_acoustic, model_encoder_audio, model_encoder_images,
                                      model_associator, model_associator1,
                                      display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate,
                                      num_classes=num_classes, num_epochs=FLAGS.num_epochs,
                                      temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
            elif FLAGS.mfcc:
                trainer = TrainerMask(model_encoder_acoustic, model_encoder_images, display_freq=FLAGS.display_freq,
                                         learning_rate=FLAGS.learning_rate, num_classes=num_classes,
                                         num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
            else:
                trainer = TrainerLoss(model_encoder_acoustic, model_encoder_audio, model_encoder_images,
                                      display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate,
                                      num_classes=num_classes, num_epochs=FLAGS.num_epochs,
                                      temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)

        else:
            trainer = TrainerNCAproxyanchor(model_encoder_acoustic, model_encoder_audio, model_encoder_images,
                              display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate,
                              num_classes=num_classes, num_epochs=FLAGS.num_epochs,
                              temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)

        if FLAGS.mode == 'train':
            checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
            if not tf.gfile.Exists(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
            # Train model
            with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/configuration.txt", "w") as outfile:

                outfile.write(
                    'Experiment: {} \nModel: {} \nLearning_rate: {}\n'.format(FLAGS.exp_name, FLAGS.model,
                                                                              FLAGS.learning_rate))
                outfile.write(
                    'Num_epochs: {} \nTotal_length: {} \nSample_length: {}\n'.format(FLAGS.num_epochs,
                                                                                     FLAGS.total_length,
                                                                                     FLAGS.sample_length))
                outfile.write(
                    'Number_of_crops: {} \nMargin: {}\nNumber of classes: {}\n'.format(FLAGS.number_of_crops,
                                                                                       FLAGS.margin, num_classes))
                outfile.write(
                    'Block_size: {} \nEmbedding: {}\nLatent weight: {}\n'.format(FLAGS.block_size, FLAGS.embedding, FLAGS.latent_loss))
                outfile.write(
                    'Train_file: {} \nValid_file: {} \nTest_file: {}\n'.format(FLAGS.train_file,
                                                                               FLAGS.valid_file,
                                                                               FLAGS.test_file))
                outfile.write(
                    'Mode: {} \nVisual_init_checkpoint: {} \nAcoustic_init_checkpoint: {} \nRestore_checkpoint: {}\n'.format(
                        FLAGS.mode,
                        FLAGS.visual_init_checkpoint,
                        FLAGS.acoustic_init_checkpoint,
                        FLAGS.restore_checkpoint))
                outfile.write('Checkpoint_dir: {} \nLog dir: {} \nBatch_size: {}\n'.format(FLAGS.checkpoint_dir,
                                                                                           FLAGS.tensorboard,
                                                                                           FLAGS.batch_size))
                outfile.write('Number of skip connections: {} \nAuto encoder: {}\nHuber: {}\nMSE: {}\n'.format(
                                                                                            FLAGS.num_skip_conn,
                                                                                           FLAGS.ae, FLAGS.huber_loss, FLAGS.MSE))
            print('{}: {} - Training started'.format(datetime.now(), FLAGS.exp_name))
            trainer.train(train_data=train_data, valid_data=valid_data)
        elif FLAGS.mode == 'test':
            # Test model
            print('{}: {} - Testing started'.format(datetime.now(), FLAGS.exp_name))
            trainer.test(test_data=test_data)
        else:
            raise ValueError('Unknown execution mode')

    else:
        with tf.device('/gpu:0'):
            if FLAGS.model == 'UNet':
                if FLAGS.encoder_type == 'Video':
                    model_encoder = UNet(input_shape=[224, 298, 3])
                elif FLAGS.encoder_type == 'Audio':
                    model_encoder = UNetSound(input_shape=[99, 257, 1])
                elif FLAGS.encoder_type == 'Ac':
                    model_encoder = UNetAc(input_shape=[36, 48, 12])
                else:
                    model_encoder = UNetE(input_shape=[36, 48, 1])
            else:#DualCamNet
                model_encoder = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=num_classes, embedding=0)
                model_encoder_images = ResNet50Model(input_shape=[224, 298, 3], num_classes=None)
                if FLAGS.num_skip_conn == 2:
                    model_encoder_acoustic = UNetAcResNet50_2skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)
                elif FLAGS.num_skip_conn == 1:
                    model_encoder_acoustic = UNetAcResNet50(input_shape=[36, 48, 12], embedding=FLAGS.ae)
                elif FLAGS.num_skip_conn == 0:
                    model_encoder_acoustic = UNetAcResNet50_0skips(input_shape=[36, 48, 12], embedding=FLAGS.ae)
        # Build trainer
        print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))

        if FLAGS.model == 'UNet':
            trainer = Trainer(model_encoder, display_freq=FLAGS.display_freq,
                              learning_rate=FLAGS.learning_rate, num_classes=num_classes,
                              num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
        else:
            if FLAGS.mfcc:
                trainer = Trainer_classification(model_encoder, display_freq=FLAGS.display_freq,
                                  learning_rate=FLAGS.learning_rate, num_classes=num_classes,
                                  num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
            else:
                trainer = Trainer_rec_class(model_encoder, model_encoder_acoustic, model_encoder_images,
                                                 display_freq=FLAGS.display_freq,
                                                 learning_rate=FLAGS.learning_rate, num_classes=num_classes,
                                                 num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling,
                                                 nr_frames=nr_frames)
        if FLAGS.mode == 'train':
            checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
            if not tf.gfile.Exists(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
            # Train model
            with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/configuration.txt", "w") as outfile:
                outfile.write('Experiment: {} \nBatch_size: {}\n Latent weight: {}\n'.format(FLAGS.exp_name,
                                                                         FLAGS.batch_size, FLAGS.latent_loss))
                outfile.write(
                    'Model: {} \nLearning_rate: {}\nNumber of classes: {}\n'.format(FLAGS.model, FLAGS.learning_rate,
                                                                                    num_classes))
                outfile.write(
                    'Num_epochs: {} \nTotal_length: {} \nSample_length: {}\n'.format(FLAGS.num_epochs,
                                                                                     FLAGS.total_length,
                                                                                     FLAGS.sample_length))
                outfile.write(
                    'Number_of_crops: {} \nCheckpoint_dir: {} \nLog dir: {}\n'.format(FLAGS.number_of_crops,
                                                                                      FLAGS.checkpoint_dir,
                                                                                      FLAGS.tensorboard))
                outfile.write(
                    'Train_file: {} \nValid_file: {} \nTest_file: {}\n'.format(FLAGS.train_file,
                                                                               FLAGS.valid_file,
                                                                               FLAGS.test_file))
                outfile.write('Number of skip connections: {} \nAuto encoder: {}\nHuber: {}\nMSE: {}\n'.format(
                    FLAGS.num_skip_conn,
                    FLAGS.ae, FLAGS.huber_loss, FLAGS.MSE))
                outfile.write(
                    'Mode: {} \nInit_checkpoint: {} \nRestore_checkpoint: {}\n'.format(FLAGS.mode,
                                                                                       FLAGS.init_checkpoint,
                                                                                       FLAGS.restore_checkpoint))
            # Train model
            print('{}: {} - Training started'.format(datetime.now(), FLAGS.exp_name))
            trainer.train(train_data=train_data, valid_data=valid_data)
        elif FLAGS.mode == 'test':
            # Test model
            print('{}: {} - Testing started'.format(datetime.now(), FLAGS.exp_name))
            trainer.test(test_data=test_data)
        else:
            raise ValueError('Unknown execution mode')


if __name__ == '__main__':
    flags.mark_flags_as_required(['mode', 'exp_name'])
    tf.app.run()
