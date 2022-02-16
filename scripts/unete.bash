CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --train_file /data/vsanguineti/tfrecords3/lists/training.txt \
--valid_file /data/vsanguineti/tfrecords3/lists/validation.txt \
--test_file /data/vsanguineti/tfrecords3/lists/testing.txt --batch_size 64 --sample_length 1 \
--total_length 1 --number_of_crops 1 --buffer_size 100 --exp_name variationallossesoutdoorenergy-3 --learning_rate 0.001  \
--tensorboard /data/vsanguineti/tensorboardaudiovideo --checkpoint_dir /data/vsanguineti/checkpointsaudiovideo/ \
 --model UNet --datatype outdoor --num_epochs 300 --num_class 10 --block_size 1 \
--probability 0 --encoder_type Energy