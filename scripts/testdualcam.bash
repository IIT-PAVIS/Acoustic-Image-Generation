#!/usr/bin/env bash
HOME_DIR="/data/"
ROOT_DIR="/home/vsanguineti/Datasets/tfrecords2/lists/"
EXP=/data/checkpointsaudiovideo/recdualcamnetunetacresnet/rec_Dualcamnet_2conn_
NUM=(4)

cd ..

for t in "${NUM[@]}"
do
MODEL_PATH=$EXP$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
echo $EPOCH_FILE

CUDA_VISIBLE_DEVICES=0 python3  saveimagesresnet.py --train_file $ROOT_DIR"/testing.txt"  \
--batch_size 16 --datatype outdoor --init_checkpoint $EPOCH_FILE --num_skip_conn 2 \
--ac_checkpoint $EPOCH_FILE

#CUDA_VISIBLE_DEVICES=0 python3 main.py --mode test --train_file $ROOT_DIR"/training.txt" --valid_file \
#$ROOT_DIR"/validation.txt" --test_file \
#$ROOT_DIR"/testing.txt" --batch_size 8 \
#--sample_length 1 --buffer_size 100 --exp_name $EXP$t --learning_rate 0.0001 --tensorboard \
#$HOME_DIR"/tensorboaraudiovideo/" --checkpoint_dir $HOME_DIR"/checkpointsaudiovideo/" --model DualCamNet \
#--datatype old --num_epochs 300 --num_class 14 --block_size 12 --mfcc 1 \
#--mfccmap 0 --restore_checkpoint $EPOCH_FILE

done