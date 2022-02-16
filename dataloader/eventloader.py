from __future__ import division
from scipy import signal
import tensorflow as tf
import numpy as np
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12

'''AVE'''

class ActionsDataLoader(object):

    def __init__(self, txt_file, mode, batch_size, sample_rate=12288, sample_length=2, embedding=1,
                 buffer_size=1, num_epochs=1, shuffle=False, normalize=False, correspondence=0,
                 random_pick=False, build_spectrogram=False, modalities=None, nr_frames=12, datakind='music'):

        # self.seed = tf.placeholder(tf.int64, shape=(), name='data_seed')   =epoch

        self.nr_frames = nr_frames
        self.txt_file = txt_file
        self.embedding = embedding
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.num_epochs = num_epochs
        self.frame_length = 246
        self.frame_step = 122# 220
        self.fft_length = 512
        self.datakind = datakind
        self.shuffle = shuffle
        self.include_audio_images = modalities is None or 0 in modalities
        self.include_audio_data = modalities is None or 1 in modalities
        self.include_video_data = modalities is None or 2 in modalities
        # correspondence = True set in training don't write in extract features
        self.correspondence = correspondence
        if self.datakind == 'music':
            self.num_channels = 13
            self.num_actions = 9
            self.num_locations = 11  # maximum number of videos for a class
        else:  # self.datakind == 'outdoor':
            self.num_channels = 12
            self.num_actions = 10
            self.num_locations = 61
        assert txt_file is not None
        assert (self.include_audio_images or self.include_audio_data or self.include_video_data) is True
        # TODO Fix this assertion to check that there are enough samples to provide the required number of crops
        # assert number_of_crops <= total_length - sample_length

        # load statistics
        # if normalize and self.include_audio_images:
        #     self.global_min, self.global_max, self.threshold = self._load_acoustic_images_stats()
        if normalize and self.include_audio_data:
            # self.global_min, self.global_max, self.global_standard_deviation,\
            #     self.global_mean= self._load_spectrogram_stats()
            self.global_standard_deviation, self.global_mean = self._load_spectrogram_stats()

        # retrieve the data from the text file
        # how many files are in file, add them to img_paths
        self._read_txt_file(mode)
        # create data set
        data = self.files.flat_map(lambda ds: tf.data.TFRecordDataset(ds, compression_type='GZIP'))

        # parse data set
        data = data.map(self._parse_sequence, num_parallel_calls=4)
        # prefetch `buffer_size` batches of elements of the dataset
        data = data.prefetch(buffer_size=buffer_size * batch_size * sample_length)

        # batch elements in groups of `sample_length` seconds
        data = data.batch(self.sample_length)
        data = data.map(self._map_function)

        # build waveform for each sequence of `sample_length`
        if self.include_audio_data:
            data = data.map(self._map_func_audio_samples_build_wav, num_parallel_calls=4)

        # build spectrogram for each waveform and normalize it
        if self.include_audio_data and build_spectrogram:
            data = data.map(self._map_func_audio_samples_build_spectrogram, num_parallel_calls=4)
        if self.include_audio_data and normalize:
            data = data.map(self._map_func_audio_samples_mean_norm, num_parallel_calls=4)

        # pick random frames from video sequence
        if random_pick:
            data = data.map(self._map_func_video_images_pick_frames, num_parallel_calls=4)
        else:
            data = data.map(self._map_func_video_images_pick_all_frames, num_parallel_calls=4)
        # for multispectral acoustic images
        # if self.include_audio_images and normalize:
        #     data = data.map(self._map_func_audio_images_piece_wise, num_parallel_calls=4)
        #     data = data.map(self._map_func_audio_images_min_max_norm, num_parallel_calls=4)

        if self.include_video_data:
            data = data.map(self._map_func_video_images, num_parallel_calls=4)
        if self.include_audio_images:
            data = data.map(self._map_func_acoustic_images, num_parallel_calls=4)
        if self.include_audio_data:
            data = data.map(self._map_func_mfcc, num_parallel_calls=4)
        if self.embedding:
            data = data.apply(tf.contrib.data.unbatch())
        if self.shuffle:
            data = data.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        # create batched dataset
        data = data.batch(batch_size)
        if self.correspondence:
            data = data.map(self._map_func_correspondence, num_parallel_calls=4)
            if mode == 'training':
                data = data.apply(tf.contrib.data.unbatch())
                data = data.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
                data = data.batch(batch_size)

        self.data = data

    def _load_spectrogram_stats(self):
        """Load spectrogram statistics."""

        stats_dir = str.join('/', self.txt_file.replace('//', '/').split('/')[:-2] + ['stats2s'])  # statsHearMean
        # min_value = np.load('{}/global_min.npy'.format(stats_dir)).clip(None, threshold_value)
        # max_value = np.load('{}/global_max.npy'.format(stats_dir)).clip(None, threshold_value)

        # global_min = tf.tile(
        #     tf.expand_dims(
        #         input=tf.expand_dims(input=tf.convert_to_tensor(min_value), axis=0),
        #         axis=0
        #     ),
        #     [500, 1, 1]
        # )
        #
        # global_max = tf.tile(
        #     tf.expand_dims(
        #         input=tf.expand_dims(input=tf.convert_to_tensor(max_value), axis=0),
        #         axis=0
        #     ),
        #     [500, 1, 1]
        # )
        mean = np.load('{}/global_mean.npy'.format(stats_dir))
        std = np.load('{}/global_std_dev.npy'.format(stats_dir))
        global_mean = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.convert_to_tensor(mean), axis=0),
                axis=0
            ),
            [200, 1, 1]
        )

        global_standard_deviation = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.convert_to_tensor(std), axis=0),
                axis=0
            ),
            [200, 1, 1]
        )

        return global_standard_deviation, global_mean  # ,global_min, global_max

    def _read_txt_file(self, mode):
        """Read the content of the text file and store it into a list."""
        num_samples = 0
        self.img_paths = []
        self.files = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            name = ''
            self.counter = 0
            for line in lines:
                img_path = line.rstrip('\n')
                path = img_path.split('/')
                n = path[-2]
                if n != name and self.counter > 0:
                    self.files2 = self.img_paths
                    if mode == 'training':
                        self.files2 = self._map_files_training(self.files2)
                    elif mode == 'validation' or mode == 'testing':
                        self.files2 = self._map_files_inference(self.files2)
                    else:
                        raise ValueError('Unknown mode')
                    if len(self.files) == 0:
                        self.files = self.files2
                    else:
                        self.files = np.concatenate((self.files, self.files2))
                    self.img_paths = []
                    num_samples += np.floor(self.counter * 1.0 / self.sample_length)

                    name = n
                    self.counter = 1
                    self.img_paths.append(img_path)
                elif n != name and self.counter == 0:
                    name = n
                    self.counter += 1
                    self.img_paths.append(img_path)
                else:
                    self.counter += 1
                    self.img_paths.append(img_path)
        # last set of samples
        self.files2 = self.img_paths
        if mode == 'training':
            self.files2 = self._map_files_training(self.files2)
        elif mode == 'validation' or mode == 'testing':
            self.files2 = self._map_files_inference(self.files2)
        else:
            raise ValueError('Unknown mode')
        if len(self.files) == 0:
            self.files = self.files2
        else:
            self.files = np.concatenate((self.files, self.files2))
        self.img_paths = []
        num_samples += np.floor(self.counter * 1.0 / self.sample_length)

        self.num_samples = (num_samples).astype(int)
        self.files = list(self.files)
        self.files = tf.convert_to_tensor(self.files, dtype=tf.string)
        self.files = tf.data.Dataset.from_tensor_slices(self.files)
        # shuffle `num_samples` blocks of files and repeat them `num_epochs`
        if self.shuffle:
            self._shuffle_and_repeat_lists(self.num_epochs, num_samples)

    def _shuffle_and_repeat_lists(self, num_epochs, num_samples):
        """Shuffle and repeat the list of paths."""
        self.files = self.files.batch(self.sample_length)
        self.files = self.files.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
        self.files = self.files.repeat(num_epochs)
        self.files = self.files.apply(tf.contrib.data.unbatch())

    def _map_files_training(self, files):
        """Input mapper for files of the training set."""
        length = self.counter
        # Compute indices of crops
        index = np.arange(length - self.sample_length + 1)
        # files_arr = np.asarray(files)
        # Crop files tensor according to the pre-computed indices
        cropped_files = []
        for ind in index:
            cropped_files.append(files[ind:ind + self.sample_length])
        cropped_files = np.stack(cropped_files, 0)
        cropped_files = np.reshape(cropped_files, (-1))
        return cropped_files

    def _map_files_inference(self, files):
        """Input mapper for files of the testing set."""
        length = self.counter
        number_of_crop = np.floor(length * 1.0 / self.sample_length).astype('int')
        index_offset = np.floor(self.sample_length).astype('int')
        # Compute indices of crops
        index = np.arange(number_of_crop)
        index = index * index_offset
        # files_arr = np.asarray(files)
        # Crop files tensor according to the pre-computed indices
        cropped_files = []
        for ind in index:
            cropped_files.append(files[ind:ind + self.sample_length])
        cropped_files = np.stack(cropped_files, 0)
        cropped_files = np.reshape(cropped_files, (-1))
        return cropped_files

    def _parse_sequence(self, sequence_example_proto):
        """Input parser for samples of the training set."""

        context_features = {'classes': tf.FixedLenFeature([], tf.int64),
                            'location': tf.FixedLenFeature([], tf.int64),
                            'event': tf.FixedLenFeature([], tf.int64),
                            }
        sequence_features = {}

        if self.include_audio_images:
            context_features.update({
                'audio_image/height': tf.FixedLenFeature([], tf.int64),
                'audio_image/width': tf.FixedLenFeature([], tf.int64),
                'audio_image/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        if self.include_audio_data:
            context_features.update({
                'audio_data/mics': tf.FixedLenFeature([], tf.int64),
                'audio_data/samples': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/data': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        if self.include_video_data:
            context_features.update({
                'video/height': tf.FixedLenFeature([], tf.int64),
                'video/width': tf.FixedLenFeature([], tf.int64),
                'video/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'video/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        # Parse single example
        parsed_context_features, parsed_sequence_features = tf.parse_single_sequence_example(sequence_example_proto,
                                                                                             context_features=context_features,
                                                                                             sequence_features=sequence_features)

        action = tf.cast(parsed_context_features['classes'], tf.int32)
        location = tf.cast(parsed_context_features['location'], tf.int32)
        event = tf.cast(parsed_context_features['event'], tf.int32)
        if self.include_audio_images:
            # Retrieve parsed context features
            audio_height = tf.cast(parsed_context_features['audio_image/height'], tf.int32)
            audio_width = tf.cast(parsed_context_features['audio_image/width'], tf.int32)
            audio_depth = tf.cast(parsed_context_features['audio_image/depth'], tf.int32)
            # Retrieve parsed audio image features
            audio_image_decoded = tf.decode_raw(parsed_sequence_features['audio/image'], tf.float32)
            # Reshape decoded audio image
            audio_image_shape = tf.stack([-1, audio_height, audio_width, audio_depth])
            audio_images = tf.reshape(audio_image_decoded, audio_image_shape)
            audio_images = tf.image.flip_left_right(audio_images)
            audio_images = tf.image.flip_up_down(audio_images)
        else:
            audio_images = tf.zeros([12, 36, 48, self.num_channels], tf.int32)

        if self.include_audio_data:
            # Retrieve parsed context features
            num_mics = tf.cast(parsed_context_features['audio_data/mics'], tf.int32)
            num_samples = tf.cast(parsed_context_features['audio_data/samples'], tf.int32)
            # Retrieve parsed audio data features
            audio_sample_decoded = tf.decode_raw(parsed_sequence_features['audio/data'], tf.int32)
            # Reshape decoded audio data
            audio_sample_shape = tf.stack([-1, num_samples])  # num_mics, num_samples
            audio_samples = tf.reshape(audio_sample_decoded, audio_sample_shape)
        else:
            audio_samples = tf.zeros([], tf.int32)

        if self.include_video_data:
            # Retrieve parsed video image features
            video_image_decoded = tf.decode_raw(parsed_sequence_features['video/image'], tf.uint8)
            # Retrieve parsed context features
            video_height = tf.cast(parsed_context_features['video/height'], tf.int32)
            video_width = tf.cast(parsed_context_features['video/width'], tf.int32)
            video_depth = tf.cast(parsed_context_features['video/depth'], tf.int32)
            # Reshape decoded video image
            video_image_shape = tf.stack([-1, video_height, video_width, video_depth])  # 224, 298, 3
            video_images = tf.reshape(video_image_decoded, video_image_shape)
        else:
            video_images = tf.zeros([], tf.int32)

        return audio_images, audio_samples, video_images, action, location, event

    def _map_function(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapping function."""

        # Convert labels into one-hot-encoded tensors
        action_encoded = tf.one_hot(
            tf.squeeze(tf.gather(action, tf.range(self.sample_length, delta=self.sample_length))), self.num_actions)
        location_encoded = tf.one_hot(
            tf.squeeze(tf.gather(location, tf.range(self.sample_length, delta=self.sample_length))), self.num_locations)

        # Reshape audio_images to be the length of a full video of `sample_length` seconds
        if self.include_audio_images:
            reshaped_audio_images = tf.reshape(audio_images, [-1, 36, 48, self.num_channels])
            #reshaped_audio_images = tf.slice(reshaped_audio_images, [0, 0, 0, 0], [-1, 36, 48, 1])
        else:
            reshaped_audio_images = tf.reshape(audio_images, [-1, 36, 48, self.num_channels])

        # Reshape audio_samples to be the length of a full video of `sample_length` seconds
        if self.include_audio_data:
            # reshaped_audio_samples = tf.reshape(audio_samples, [-1, _NUMBER_OF_MICS, _NUMBER_OF_SAMPLES])
            reshaped_audio_samples = tf.reshape(audio_samples, [-1, 1, _NUMBER_OF_SAMPLES])
        else:
            reshaped_audio_samples = tf.zeros([], tf.int32)

        # Reshape audio_samples to be the length of a full video of `sample_length` seconds
        if self.include_video_data:
            # reshaped_video_images = tf.reshape(video_images, [-1, 480, 640, 3])
            reshaped_video_images = tf.reshape(video_images, [-1, 224, 298, 3])  # 224, 298, 3
        else:
            reshaped_video_images = tf.zeros([], tf.int32)

        return reshaped_audio_images, reshaped_audio_samples, reshaped_video_images, action_encoded, location_encoded, event

    def _map_func_video_images_pick_all_frames(self, audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event):
        """Pick nr_frames random frames."""
        #in 1 second 12
        if self.embedding:
            action = tf.expand_dims(action, 0)
            action = tf.tile(action, (1, self.nr_frames))
            action = tf.reshape(action, (-1, self.num_actions))
            location = tf.expand_dims(location, 0)
            location = tf.tile(location, (1, self.nr_frames))
            location = tf.reshape(location, (-1, self.num_locations))
            event = tf.expand_dims(event, 0)
            event = tf.tile(event, (1, self.nr_frames))
            event = tf.reshape(event, (-1, 1))
        return audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event

    def _map_func_video_images_pick_frames(self, audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event):
        """Pick nr_frames random frames."""
        # in 1 second 12
        # pick one random  image if 1 s blocksize 1
        # selected_video_images = self._pick_random_frames(video_images)
        # #make then mean of 12 mfcc and 12 acoustic log energy
        # selected_audio_samples, selected_audio_images = self._pick_mean(audio_samples, audio_images)
        selected_video_images, selected_audio_samples, filtered_selected_audio_samples, selected_audio_images = \
            self._pick_random_frames(video_images, audio_samples, filtered_audio_samples, audio_images)
        action = tf.expand_dims(action, 0)
        action = tf.tile(action, (1, self.nr_frames))
        action = tf.reshape(action, (-1, self.num_actions))
        location = tf.expand_dims(location, 0)
        location = tf.tile(location, (1, self.nr_frames))
        location = tf.reshape(location, (-1, self.num_locations))
        return selected_audio_images, selected_audio_samples, selected_video_images, action, location, filtered_selected_audio_samples, event

    def _pick_random_frames(self, samples, audio_samples, filtered_audio_samples, audio_images, event):
        num_frames = tf.shape(samples)[0]  # how many images
        n_to_sample = tf.constant([self.nr_frames])  # how many to keep, 5 in temporal resnet 50, 8 in resnet18
        mask = self._sample_mask(num_frames, n_to_sample)  # pick n_to_sample
        frames = tf.boolean_mask(samples, mask)  # keep element in ones position
        filtered_framesaudio = tf.boolean_mask(filtered_audio_samples, mask)
        framesaudio = tf.boolean_mask(audio_samples, mask)
        framesaudioimage = tf.boolean_mask(audio_images, mask)
        return frames, framesaudio, filtered_framesaudio, framesaudioimage, event

    def _sample_mask(self, num_frames, sample_size):
        # randomly choose between uniform or random sampling
        end = tf.subtract(num_frames, 1)  # last index
        indexes = tf.to_int32(tf.linspace(
            0.0, tf.to_float(end), sample_size[0]))  # uses linspace to draw uniformly samples between 0 and end
        # find indexes
        updates = tf.ones(sample_size, dtype=tf.int32)  # ones
        mask = tf.scatter_nd(tf.expand_dims(indexes, 1),
                             updates, tf.expand_dims(num_frames, 0))  # put ones in indexes positions

        compare = tf.ones([num_frames], dtype=tf.int32)  # ones in all num_frames
        mask = tf.equal(mask, compare)  # see where are ones
        return mask

    def _pick_mean(self, samples, audioimages):
        num_frames = self.sample_length*12  # how many images
        n_to_sample = self.nr_frames# how many to keep, 5 in temporal resnet 50, 8 in resnet18
        # pick nr_frames
        frames, audioimages = self._sample_mean(samples, audioimages, num_frames, n_to_sample)
        return frames, audioimages

    def _sample_mean(self, samples, audioimages, num_frames, sample_size):

        surrounding_frames = 3
        endvalue = num_frames - 1 - surrounding_frames
        # random center according to how many
        # offset = surrounding_frames
        offset = tf.random_uniform(shape=tf.shape(num_frames), dtype=tf.int32, minval=surrounding_frames,
                                   maxval=endvalue)
        end = offset + surrounding_frames
        start = offset - surrounding_frames
        number = end - start + 1

        # uses linspace to draw sample from consecutive samples between offset and end
        indexes = tf.cast(tf.linspace(
            tf.to_float(start), tf.to_float(end), number), tf.int32)

        # find indexes
        updates = tf.ones(number, dtype=tf.int32)  # ones
        mask1 = tf.scatter_nd(tf.expand_dims(indexes, 1),
                              updates, tf.expand_dims(num_frames, 0))  # put ones in indexes positions

        compare = tf.ones([num_frames], dtype=tf.int32)  # ones in all num_frames
        mask = tf.equal(mask1, compare)  # see where are ones

        frames = tf.boolean_mask(samples, mask)  # keep element in ones position
        frames = tf.reduce_mean(frames, 0, keep_dims=True)

        framesimages = tf.boolean_mask(audioimages, mask)  # keep element in ones position
        framesimages = tf.reduce_mean(framesimages, 0, keep_dims=True)

        return frames, framesimages

    def _map_func_audio_samples_build_wav(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapper to build waveform audio from raw audio samples."""

        audio_wav = tf.reshape(audio_samples, (-1, _NUMBER_OF_SAMPLES))
        filtered_wav = tf.py_func(self.butter_lowpass_filter, [audio_wav], tf.float32)
        return audio_images, audio_wav, video_images, action, location, filtered_wav, event

    def butter_lowpass(self, cutoff=125, order=10):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff=125, order=10):
        b, a = self.butter_lowpass(cutoff, order=order)
        y = signal.filtfilt(b, a, data)
        y = np.float32(y)
        return y

    # def _map_func_audio_samples_build_wav(self, audio_images, audio_samples, video_images, action, location):
    #     """Input mapper to build waveform audio from raw audio samples."""
    #
    #     audio_wav = tf.py_func(self._build_wav_py_function, [audio_samples], tf.float32)
    #
    #     return audio_images, audio_wav, video_images, action, location
    #
    # def _build_wav_py_function(self, audio_data):
    #     """Python function to build a waveform audio from audio samples."""
    #     audio_data = audio_data.astype(np.float32)
    #     # concatenate 1/12
    #     audio_data = audio_data.flatten('C')
    #
    #     # audio_data = audio_data / abs(
    #     #     max(audio_data.min(), audio_data.max(), key=abs))
    #     # size is correct because we take at least one second of data
    #     # Re-sample audio to 22 kHz
    #     # audio_wav = librosa.core.resample(audio_data, audio_data.shape[0] / self.sample_length,
    #     #                                   self.sample_rate)
    #     audio_wav = audio_data
    #     # range between -1 and 1
    #     audio_wav = audio_wav / abs(max(audio_wav.min(), audio_wav.max(), key=abs))
    #     # Make range [-256, 256]
    #     #audio_wav *= 256.0
    #
    #     return audio_wav

    def _map_func_audio_images_piece_wise(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapper to apply piece-wise normalization to the audio images."""

        audio_images_norm = tf.where(tf.greater(audio_images, self.threshold), self.threshold, audio_images)

        return audio_images_norm, audio_samples, video_images, action, location, event

    def _map_func_audio_images_min_max_norm(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapper to apply min-max normalization to the audio images."""

        audio_images_norm = tf.divide(tf.subtract(audio_images, self.global_min),
                                      tf.subtract(self.global_max, self.global_min))

        return audio_images_norm, audio_samples, video_images, action, location, event

    def _map_func_audio_samples_min_max_norm(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapper to apply min-max normalization to the audio samples."""

        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_min),
                                       tf.subtract(self.global_max, self.global_min))

        return audio_images, audio_samples_norm, video_images, action, location, event

    def _map_func_audio_samples_mean_norm(self, audio_images, audio_samples, video_images, action, location, event):
        """Input mapper to apply min-max normalization to the audio samples."""

        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_mean), self.global_standard_deviation)

        return audio_images, audio_samples_norm, video_images, action, location, event

    def _map_func_video_images(self, audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event):
        """Input mapper to pre-processes the given image for training."""

        # function to rescale image, normalizing subtracting the mean and takes random crops in different positions
        # and to flip right left images for is_training=True for augmenting data
        # so we leave is_training=false only to take central crop
        def prepare_image(image):
            # return vgg_preprocessing.preprocess_image(image, _IMAGE_SIZE, _IMAGE_SIZE, is_training=False)
            # return self._aspect_preserving_resize(image, _IMAGE_SIZE)
            return self._normalize_images_rescaled(image)

        processed_images = tf.map_fn(prepare_image, video_images, dtype=tf.float32, back_prop=False)

        return audio_images, audio_samples, processed_images, action, location, filtered_audio_samples, event

    def _normalize_images_rescaled(self, image):
        image.set_shape([224, 298, 3])
        image = tf.to_float(image)
        #image = self._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image = image[..., ::-1]
        image = image * (1. / 255.)
        return image

    def _map_func_acoustic_images(self, audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event):
        """Input mapper to pre-processes the given image for training."""

        # function to rescale image, normalizing subtracting the mean and takes random crops in different positions
        # and to flip right left images for is_training=True for augmenting data
        # so we leave is_training=false only to take central crop
        def prepare_image(image):
            # return vgg_preprocessing.preprocess_image(image, _IMAGE_SIZE, _IMAGE_SIZE, is_training=False)
            # return self._aspect_preserving_resize(image, _IMAGE_SIZE)
            return self._normalize_acoustic_images_rescaled(image)

        processed_images = tf.map_fn(prepare_image, audio_images, dtype=tf.float32, back_prop=False)

        return processed_images, audio_samples, video_images, action, location, filtered_audio_samples, event

    def _normalize_acoustic_images_rescaled(self, image):
        image.set_shape([36, 48, 12])
        image = tf.to_float(image)
        imagemin = tf.reduce_min(image, axis=[0, 1, 2], keep_dims=True)
        image = image - imagemin
        imagemax = tf.reduce_max(image, axis=[0, 1, 2], keep_dims=True)
        image = image/imagemax
        return image

    def _map_func_mfcc(self, audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event):
        """Input mapper to pre-processes the given image for training."""

        # function to rescale image, normalizing subtracting the mean and takes random crops in different positions
        # and to flip right left images for is_training=True for augmenting data
        # so we leave is_training=false only to take central crop
        def prepare_audio(audio):
            # return vgg_preprocessing.preprocess_image(image, _IMAGE_SIZE, _IMAGE_SIZE, is_training=False)
            # return self._aspect_preserving_resize(image, _IMAGE_SIZE)
            return self._normalize_mfcc(audio)

        audio_samples = tf.map_fn(prepare_audio, audio_samples, dtype=tf.float32, back_prop=False)
        filtered_audio_samples = tf.map_fn(prepare_audio, filtered_audio_samples, dtype=tf.float32, back_prop=False)
        return audio_images, audio_samples, video_images, action, location, filtered_audio_samples, event

    def _normalize_mfcc(self, mfcc):
        mfcc.set_shape([12])
        image = tf.to_float(mfcc)
        imagemin = tf.reduce_min(image, axis=[0], keep_dims=True)
        image = image - imagemin
        imagemax = tf.reduce_max(image, axis=[0], keep_dims=True)
        image = image/imagemax
        return image

    def _aspect_preserving_resize(self, image, smallest_side):
        """Resize images preserving the original aspect ratio.
        Args:
          image: A 3-D image `Tensor`.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
        Returns:
          resized_image: A 3-D tensor containing the resized image.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])

        return resized_image

    def _smallest_size_at_least(self, height, width, smallest_side):
        """Computes new shape with the smallest side equal to `smallest_side`.
        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.
        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: and int32 scalar tensor indicating the new width.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / height,
                        lambda: smallest_side / width
                        )
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        return new_height, new_width

    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = _mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.
        Returns:
          the centered image.
        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(image, num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= means[i]

        return tf.concat(channels, axis=2)


    def _map_func_audio_samples_build_spectrogram(self, audio_images, audio_wav, processed_images, action, location, filtered_audio_wav, event):
        """Input mapper to build waveform audio from raw audio samples."""

        magnitude_spectrograms = tf.py_func(self._build_spectrograms_function, [audio_wav], tf.float32)
        magnitude_spectrograms = tf.reshape(magnitude_spectrograms, (self.sample_length*_FRAMES_PER_SECOND, self.num_channels))

        filtered_magnitude_spectrograms = tf.py_func(self._build_spectrograms_function, [filtered_audio_wav], tf.float32)
        filtered_magnitude_spectrograms = tf.reshape(filtered_magnitude_spectrograms, (self.sample_length*_FRAMES_PER_SECOND, self.num_channels))

        return audio_images, magnitude_spectrograms, processed_images, action, location, filtered_magnitude_spectrograms, event

    def _build_spectrograms_function(self, audio_data):

        n = np.shape(audio_data)[0]
        window = signal.tukey(1024, alpha=0.75)
        window = np.tile(window, (n, 1))
        window = np.reshape(window, (n, _NUMBER_OF_SAMPLES))
        raw_audio = audio_data * window
        fftdata = np.abs(np.fft.rfft(raw_audio, 1024, axis=1))[:, :-1]
        fftdata = fftdata ** 2
        # energy = np.sum(fftdata, axis=-1)
        lifter_num = 22
        lo_freq = 0
        hi_freq = 6400
        filter_num = 24
        mfcc_num = 12
        fft_len = 512

        dct_base = np.zeros((filter_num, mfcc_num))
        for m in range(mfcc_num):
            dct_base[:, m] = np.cos((m + 1) * np.pi / filter_num * (np.arange(filter_num) + 0.5))
        lifter = 1 + (lifter_num / 2) * np.sin(np.pi * (1 + np.arange(mfcc_num)) / lifter_num)

        mfnorm = np.sqrt(2.0 / filter_num)

        filter_mat = self.createfilters(fft_len, filter_num, lo_freq, hi_freq, 2*hi_freq)
        coefficients = self.get_feats(fft_len, fftdata, mfcc_num, dct_base, mfnorm, lifter, filter_mat)
        # coefficients[:, 0] = energy
        coefficients = np.float32(coefficients)
        return coefficients

    def createfilters(self, fft_len, filter_num, lo_freq, hi_freq, samp_freq):

        filter_mat = np.zeros((fft_len, filter_num))

        mel2freq = lambda mel: 700.0 * (np.exp(mel / 1127.0) - 1)
        freq2mel = lambda freq: 1127 * (np.log(1 + (freq / 700.0)))

        lo_mel = freq2mel(lo_freq)
        hi_mel = freq2mel(hi_freq)

        mel_c = np.linspace(lo_mel, hi_mel, filter_num + 2)
        freq_c = mel2freq(mel_c)
        # freq_c = np.linspace(lo_freq, hi_freq, filter_num + 2)
        point_c = (freq_c / float(samp_freq) * (fft_len - 1) * 2)
        point_c = np.floor(point_c).astype('int')

        for f in range(filter_num):
            d1 = point_c[f + 1] - point_c[f]
            d2 = point_c[f + 2] - point_c[f + 1]

            filter_mat[point_c[f]:point_c[f + 1] + 1, f] = np.linspace(0, 1, d1 + 1)
            filter_mat[point_c[f + 1]:point_c[f + 2] + 1, f] = np.linspace(1, 0, d2 + 1)

        return filter_mat

    def get_feats(self, fft_len, beam, mfcc_num, dct_base, mfnorm, lifter, filter_mat):
        n = beam.shape[0]
        beam = np.reshape(beam, [n, fft_len])
        # filters
        melspec = np.dot(beam, filter_mat)

        # floor (before log)
        melspec[melspec < 0.001] = 0.001

        # log
        melspec = np.log(melspec)

        # dct
        mfcc_coefficients = np.dot(melspec, dct_base)
        mfcc_coefficients *= mfnorm

        # lifter
        mfcc_coefficients *= lifter

        # sane fixes
        mfcc_coefficients[np.isnan(mfcc_coefficients)] = 0
        mfcc_coefficients[np.isinf(mfcc_coefficients)] = 0

        coefficients = np.reshape(mfcc_coefficients, [n, mfcc_num])

        return coefficients

    def _map_func_audio_samples_build_spectrogram2(self, audio_images, audio_wav, processed_images, action, location, event):
        """Input mapper to build spectrogram from waveform audio."""

        audio_stfts = tf.contrib.signal.stft(audio_wav, frame_length=self.frame_length,
                                             frame_step=self.frame_step, fft_length=self.fft_length)

        magnitude_spectrograms = tf.abs(audio_stfts)#tf.expand_dims(tf.abs(audio_stfts), 1)

        return audio_images, magnitude_spectrograms, processed_images, action, location, event

    def _map_func_correspondence(self, audio_images, magnitude_spectrograms, processed_images, action, location, filtered_magnitude_spectrograms, event):
        #for each data we have copied silence mfcc, we tile to obtain corresponding acoustic images and double data
        action = tf.argmax(action, axis=1)
        location = tf.argmax(location, axis=1)
        mfcc = filtered_magnitude_spectrograms - tf.reduce_min(filtered_magnitude_spectrograms, axis=[1], keep_dims=True)
        mfcc = mfcc / tf.reduce_max(mfcc, axis=[1], keep_dims=True)
        mfccmap = tf.reshape(mfcc, (-1, 1, 12))
        mfccmap = tf.tile(mfccmap, (1, 36 * 48, 1))
        mfccmap = tf.reshape(mfccmap, (-1, 36, 48, 12))
        audio_images2 = mfccmap

        # concatenate two times
        reshaped_video_images = tf.concat((processed_images, processed_images), 0)
        videoaction = tf.concat((action, action), 0)
        videolocation = tf.concat((location, location), 0)
        audio_images = tf.concat((audio_images, audio_images2), 0)
        reshaped_audio_samples = tf.concat((magnitude_spectrograms, filtered_magnitude_spectrograms), 0)

        # correspondence labels
        # first part ones second zero
        labels = tf.concat((tf.ones((tf.shape(processed_images)[0]), dtype=tf.int32),
                            tf.zeros((tf.shape(processed_images)[0]), dtype=tf.int32)), 0)
        # one hot encoding
        labels = tf.one_hot(labels, 2)
        videoaction = tf.one_hot(videoaction, self.num_actions)
        videolocation = tf.one_hot(videolocation, self.num_locations)

        # shuffle with same seed in the same order
        # audio_images = tf.random_shuffle(audio_images, seed=0)
        # videoaction = tf.random_shuffle(videoaction, seed=0)
        # videolocation = tf.random_shuffle(videolocation, seed=0)
        # audioaction = tf.random_shuffle(audioaction, seed=0)
        # audiolocation = tf.random_shuffle(audiolocation, seed=0)
        # reshaped_video_images = tf.random_shuffle(reshaped_video_images, seed=0)
        # reshaped_audio_samples = tf.random_shuffle(reshaped_audio_samples, seed=0)
        # labels = tf.random_shuffle(labels, seed=0)

        return audio_images, reshaped_audio_samples, reshaped_video_images, videoaction, \
               videolocation, labels, event

    @property
    def total_batches(self):
        total_batches = int(math.ceil(self.num_samples / self.batch_size))
        return total_batches
