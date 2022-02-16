#!/usr/bin/env bash
HOME_DIR="/data/vsanguineti"
ROOT_DIR="/data/vsanguineti/tfrecords2/lists2/"
EXP="rec_Dualcamnet_2conn_"
ST=(1 2 3 4)
P[0]=0
P[1]=0
i=0
EXP2=$HOME_DIR"/checkpointsaudiovideo/Unetacresnetnoreg_outdoor_"
for t in "${ST[@]}"
do
MODEL_PATH=$EXP2$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
#unetacresnet with frames
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train \
--train_file  $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt" \
 --batch_size 32 --sample_length 1 --buffer_size 100 \
--exp_name $EXP$t --learning_rate 0.0001 --tensorboard $HOME_DIR"/tensorboardaudiovideo/" \
--checkpoint_dir $HOME_DIR"/checkpointsaudiovideo/" --model DualCamNet --datatype outdoor --num_epochs 100 \
--num_class 10 --num_skip_conn 2 --block_size 12 --init_checkpoint $EPOCH_FILE
done