CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --train_file /home/vsanguineti/Datasets/tfrecords3/lists/trainingfile.txt \
--valid_file /home/vsanguineti/Datasets/tfrecords3/lists/trainingfile.txt \
--test_file /home/vsanguineti/Datasets/tfrecords3/lists/trainingfile.txt --batch_size 16 --sample_length 1 \
 --total_length 1 --number_of_crops 1 --buffer_size 100 --exp_name variationall1outdoorlogfusion --learning_rate 0.0008 \
--tensorboard /data/tensorboard/ --checkpoint_dir /data/checkpoints/ --model UNet  --datatype outdoor --num_epochs 300 \
 --num_class 10 --block_size 1 --probability 0