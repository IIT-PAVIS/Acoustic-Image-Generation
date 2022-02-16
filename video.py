from scipy import signal
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import scipy.io.wavfile
import subprocess

_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FPS = 12

'''Create audio wav'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Data directory containing synchronized audio/video frames', type=str)
    parser.add_argument('out_dir', help='Output directory where to store generated video', type=str)
    parser.add_argument('mic_id', help='Index of the microphone to use', type=int)
    parsed_args = parser.parse_args()

    data_dir = parsed_args.data_dir
    out_dir = parsed_args.out_dir
    mic_id = parsed_args.mic_id

    audio_dir = '{}/audio'.format(data_dir)
    video_dir = '{}/beam_matlab'.format(data_dir)

    num_files = len([name for name in os.listdir(audio_dir) if name.endswith('.dc')])

    audio_data = np.zeros((num_files, _NUMBER_OF_MICS, _NUMBER_OF_SAMPLES), dtype=np.float32)

    print('Reading audio data from directory {} and microphone {}'.format(audio_dir, mic_id))

    for h in range(0, num_files):
        # Compose audio file name
        audio_sample_file = '{}/A_{:06d}.dc'.format(audio_dir, h + 1)

        # Read audo file
        with open(audio_sample_file) as fid:
            audio_data_mic = np.fromfile(fid, np.int32).reshape((_NUMBER_OF_MICS, _NUMBER_OF_SAMPLES), order='F')
            audio_data[h, :, :] = audio_data_mic

    print('Extracting microphone data')

    audio_data_mic = audio_data[:, mic_id, :]
    audio_data_mic_flat = audio_data_mic.flatten('C')
    audio_data_mic_norm = audio_data_mic_flat / abs(max(audio_data_mic_flat.min(), audio_data_mic_flat.max(), key=abs))

    print('Creating audio track')

    audio_file = '{}/audio_track2.wav'.format(out_dir)
    scipy.io.wavfile.write('{}'.format(audio_file), _FPS * 1000, audio_data_mic_norm)

    plt.figure(figsize=(20, 10))
    plt.plot(audio_data_mic_norm)
    plt.axis('off')
    plt.show()

    # print('Creating video track')
    #
    # video_file = '{}/video_track.avi'.format(out_dir)
    # command = 'ffmpeg -y -r {} -f image2 -s 640x480 -i {}/I_%06d.bmp -vcodec libx264 -crf 25 -pix_fmt yuv420p {}'.format(_FPS, video_dir.replace(' ', '\ '), video_file.replace(' ', '\ '))
    # exit_code = subprocess.call(command, shell=True)
    #
    # if exit_code:
    #     print('Failed')
    #     exit(1)
    # else:
    #     print('Done')
    #
    # print('Merging audio and video tracks')
    #
    # command = 'ffmpeg -y -i {} -i {} -codec copy -shortest {}/video.avi'.format(audio_file.replace(' ', '\ '), video_file.replace(' ', '\ '), out_dir.replace(' ', '\ '))
    # exit_code = subprocess.call(command, shell=True)
    #
    # if exit_code:
    #     print('Failed')
    #     exit(1)
    # else:
    #     print('Done')
    #
    # print('Cleaning temporary files')
    #
    # try:
    #     os.remove(audio_file)
    #     os.remove(video_file)
    # except OSError as e:
    #     print('An unexpected error occurred while remove temporary audio and video track files. {}', e)
    #
    # print('Done')
