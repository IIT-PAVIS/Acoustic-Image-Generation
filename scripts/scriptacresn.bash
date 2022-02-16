#unetacresnet with frames with silence
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train \
--train_file /data/vsanguineti/acviwtfrecords/lists/training.txt \
--valid_file /data/vsanguineti/acviwtfrecords/lists/validation.txt \
--test_file /data/vsanguineti/acviwtfrecords/lists/testing.txt --batch_size 64 --sample_length 1 --buffer_size 100 \
--exp_name UNetacviwonosound --learning_rate 0.0001 --tensorboard /data/vsanguineti/tensorboardaudiovideo/ \
 --checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ --model UNet --datatype outdoor --num_epochs 300 \
  --num_class 10 --block_size 12 --embedding 1 --mfcc 1 --correspondence 1 --latent_loss 0.000001 --visual_init_checkpoint /data/vsanguineti/checkpoints/resnet/resnet_v1_50.ckpt

#unetacresnet with frames
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train \
--train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 64 --sample_length 1 --buffer_size 100 \
--exp_name UnetResAcframestwoconcconvloss6 --learning_rate 0.0001 --tensorboard /data/vsanguineti/tensorboardaudiovideo/ \
--checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ --model UNet --datatype outdoor --num_epochs 300 \
--num_class 10 --block_size 12 --embedding 1 --mfcc 1 --latent_loss 0.000001 --visual_init_checkpoint /data/vsanguineti/checkpoints/resnet/resnet_v1_50.ckpt

CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train \
--train_file /home/vsanguineti/Datasets/acviwtfrecords/lists/training.txt \
--valid_file /home/vsanguineti/Datasets/acviwtfrecords/lists/validation.txt \
--test_file /home/vsanguineti/Datasets/acviwtfrecords/lists/testing.txt --batch_size 64 --sample_length 1 --buffer_size 100 \
--exp_name UNetacviw --learning_rate 0.0001 --tensorboard /data/tensorboard/ \
 --checkpoint_dir /data/checkpoints/ --model UNet --datatype outdoor --num_epochs 300 \
 --num_class 10 --block_size 12 --embedding 1 --mfcc 1 --latent_loss 0.000001 --visual_init_checkpoint /home/vsanguineti/checkpoints/resnet/resnet_v1_50.ckpt

#dualcamnet mfccmap

CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train \
--train_file /home/vsanguineti/Datasets/acviwtfrecords/lists/training.txt \
--valid_file /home/vsanguineti/Datasets/acviwtfrecords/lists/validation.txt \
--test_file /home/vsanguineti/Datasets/acviwtfrecords/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
--exp_name DualCamNetmfccmapout --learning_rate 0.0001 --tensorboard /data/tensorboard/ --checkpoint_dir /data/checkpoints/ \
--model DualCamNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 12 --mfcc 1 --mfccmap 1

CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train \
 --train_file /home/vsanguineti/Datasets/dualcam_actions_dataset/1_second/lists/training.txt \
 --valid_file /home/vsanguineti/Datasets/dualcam_actions_dataset/1_second/lists/validation.txt \
 --test_file /home/vsanguineti/Datasets/dualcam_actions_dataset/1_second/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
 --exp_name DualCamNetmfccmapoldaction --learning_rate 0.0001 --tensorboard /data/tensorboard/ --checkpoint_dir /data/checkpoints/ \
 --model DualCamNet --datatype old --num_epochs 300 --num_class 14 --block_size 12 --mfcc 1 --mfccmap 1

#dualcamnet
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train \
--train_file /home/vsanguineti/Datasets/tfrecords2/lists/training.txt \
--valid_file /home/vsanguineti/Datasets/tfrecords2/lists/validation.txt \
--test_file /home/vsanguineti/Datasets/tfrecords2/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
--exp_name DualCamNet --learning_rate 0.0001 --tensorboard /data/tensorboard/ --checkpoint_dir /data/checkpoints/ \
--model DualCamNet --datatype outdoor --num_epochs 100 --num_class 10 --block_size 1 --mfcc 1

#dualcamnet with unetacresnet

CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --train_file /home/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /home/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /home/vsanguineti/tfrecords2/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
--exp_name DualCamNetframesskipmore --learning_rate 0.0001 --tensorboard /home/vsanguineti/tensorboardaudiovideo/ \
--checkpoint_dir /home/vsanguineti/checkpointsaudiovideo/ --model DualCamNet --datatype outdoor --num_epochs 100 \
--num_class 10 --block_size 12 --embedding 0 \
--init_checkpoint /home/vsanguineti/checkpointsaudiovideo/UnetResAcframestwoconcconvloss6/epoch_15.ckpt

CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --train_file /home/vsanguineti/acviwtfrecords/lists/training.txt \
--valid_file /home/vsanguineti/acviwtfrecords/lists/validation.txt \
--test_file /home/vsanguineti/acviwtfrecords/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
--exp_name DualCamNetrecf0normalizeupdateops --learning_rate 0.0001 --tensorboard /home/vsanguineti/tensorboardaudiovideo/ \
--checkpoint_dir /home/vsanguineti/checkpointsaudiovideo/ --model DualCamNet --datatype outdoor --num_epochs 100 \
--num_class 10 --block_size 12 --embedding 0 \
--init_checkpoint /home/vsanguineti/checkpointsaudiovideo/UnetResAcframestwoconcconvloss6-f/epoch_15.ckpt

 CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --train_file /home/vsanguineti/dualcam_actions_dataset/1_second/lists/training.txt \
 --valid_file /home/vsanguineti/dualcam_actions_dataset/1_second/lists/validation.txt \
 --test_file /home/vsanguineti/dualcam_actions_dataset/1_second/lists/testing.txt --batch_size 32 --sample_length 1 --buffer_size 100 \
 --exp_name DualCamNetactionrec1second --learning_rate 0.0001 --tensorboard /home/vsanguineti/tensorboardaudiovideo/ \
 --checkpoint_dir /home/vsanguineti/checkpointsaudiovideo/ --model DualCamNet --datatype old --num_epochs 300 \
 --num_class 14 --block_size 12 --embedding 0 \
 --init_checkpoint /home/vsanguineti/checkpointsaudiovideo/UnetResAcframesaction1secondinizializza/epoch_30.ckpt

