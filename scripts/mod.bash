#fusion --embedding 1 --jointmvae 1
python3 main.py --mode train --train_file /home/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /home/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /home/vsanguineti/tfrecords2/lists/testing.txt --batch_size 32 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variationaljoint10-4lessvar --proxy 0 --learning_rate 0.0001 \
--tensorboard /home/vsanguineti/tensorboardaudiovideo/ --checkpoint_dir /home/vsanguineti/checkpointsaudiovideo/ \
--model UNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --jointmvae 1 \
--acoustic_init_checkpoint /home/vsanguineti/checkpointsaudiovideo/variationalachuge/epoch_88.ckpt \
--audio_init_checkpoint /home/vsanguineti/checkpointsaudiovideo/variationalaudiohuge-4/epoch_74.ckpt \
--visual_init_checkpoint /home/vsanguineti/checkpointsaudiovideo/variationalvideohuge-4/epoch_108.ckpt

#moddrop --embedding 1 --moddrop 1 --jointmvae 1
python3 main.py --mode train --train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 32 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variationaljoint50p --proxy 0 --learning_rate 0.0001 \
--tensorboard /data/vsanguineti/tensorboardaudiovideo/ --checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ \
--model UNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --moddrop 1 --jointmvae 1 \
 --init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationaljoint10-4lessvar/epoch_138.ckpt

#feature --embedding 1 --jointmvae 1 --onlyaudiovideo 1
python3 main.py --mode train --train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 64 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variationaljointtwoae5 --proxy 0 --learning_rate 0.00001 \
--tensorboard /data/vsanguineti/tensorboardaudiovideo/ --checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ \
--model UNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --jointmvae 1 \
--onlyaudiovideo 1 --init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationaljoint10-4lessvar/epoch_111.ckpt

#twofromscratch --embedding 1 --jointmvae 1 --fusion 1
python3 main.py --mode train --train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 32 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variationaljointtwozero-5 --proxy 0 --learning_rate 0.00001 \
--tensorboard /data/vsanguineti/tensorboardaudiovideo/ --checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ \
--model UNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --fusion 1 --jointmvae 1 \
--acoustic_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationalachuge/epoch_201.ckpt \
--audio_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationalaudiohuge-4/epoch_173.ckpt \
--visual_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationalvideohuge-4/epoch_108.ckpt

#associator l2 --embedding 1 --fusion 1 --l2 1 --project 1
python3 main.py --mode train --train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 64 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variational10-4 --proxy 0 \
--learning_rate 0.0001 --tensorboard /data/vsanguineti/tensorboardaudiovideo/ \
--checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ --model UNet --datatype outdoor \
--num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --fusion 1 --l2 1 \
--acoustic_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationallogvarac/epoch_284.ckpt \
--audio_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationallogvaraudio/epoch_92.ckpt \
--visual_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationalvideo-4/epoch_93.ckpt --project 1

#associator triplet --embedding 1 --fusion 1 --project 1
python3 main.py --mode train --train_file /data/vsanguineti/tfrecords2/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords2/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords2/lists/testing.txt --batch_size 64 --sample_length 1 --total_length 1 \
--number_of_crops 1 --buffer_size 100 --exp_name variationaltriplet10-4 --proxy 0 \
--learning_rate 0.0001 --tensorboard /data/vsanguineti/tensorboardaudiovideo/ \
--checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ --model UNet --datatype outdoor \
--num_epochs 300 --num_class 10 --block_size 1 --embedding 1 --fusion 1 --l2 0 \
--acoustic_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationallogvarac/epoch_284.ckpt \
--audio_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationallogvaraudio/epoch_92.ckpt \
--visual_init_checkpoint /data/vsanguineti/checkpointsaudiovideo/variationalvideo-4/epoch_93.ckpt --project 1