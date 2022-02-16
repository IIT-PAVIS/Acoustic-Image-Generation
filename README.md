# Acoustic-Image-Generation
Code for the paper "Audio-Visual Localization by Acoustic Image Generation", AAAI 2021

## Requirements

- Python 3
- TensorFlow 1.14.0 >=

# Contents

We provide several scripts and a package with all the necessary code for training and testing our model. The code is organized in several folders and a couple of main scripts as follows:

- The `main` script is used for training and testing the different models: UNet, DualCamNet with real and generated images.

- The `showimages_bb` plots FlickrSoundnet energy from a UNet checkpoint and list of tfrecords of FlickrSoundnet.

- The `showimages` plots ACIVW and AVIA energy from a UNet checkpoint and list of testing tfrecords.

- The `showvideo` plots VGG Sound energy from a UNet checkpoint and list of tfrecords of a video.

- The `meanstd` computes the mean the computed metrics of 5 experiments excluding for each metric min and max values and saves them in a xlsx file.
