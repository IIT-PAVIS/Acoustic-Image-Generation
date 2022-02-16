#!/usr/bin/env bash
HOME_DIR="/data/vsanguineti"
ROOT_DIR="/data/vsanguineti/tfrecords2/lists/"
EXP="rec_Dualcamnet_noconn_"
ST=(2 4)
P[0]=0
P[1]=0
i=0
EXP2=$HOME_DIR"/checkpointsaudiovideo/Unetacresnetnoconn_outdoor_"
for t in "${ST[@]}"
do
MODEL_PATH=$EXP2$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
#unetacresnet with frames
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train \
--train_file  $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt" \
 --batch_size 32 --sample_length 1 --buffer_size 50 \
--exp_name $EXP$t --learning_rate 0.0001 --tensorboard $HOME_DIR"/tensorboardaudiovideo/" \
--checkpoint_dir $HOME_DIR"/checkpointsaudiovideo/" --model DualCamNet --datatype outdoor --num_epochs 100 \
--num_class 10 --num_skip_conn 0 --block_size 12 --init_checkpoint $EPOCH_FILE
done

# !/usr/bin/env bash
#HOME_DIR="/data/vsanguineti/"
#ROOT_DIR="/data/vsanguineti/dualcam_actions_dataset2/1_second/lists/"
#EXP="Unetacresnet1conn_actionsmoreepochs10-5_"
#ST=(1 2 3 4 5)
#P[0]=0
#P[1]=0
#i=0
#
#for t in "${ST[@]}"
#do
##unetacresnet with frames
#CUDA_VISIBLE_DEVICES=$i python3 main.py --mode train \
#--train_file  $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt" \
# --batch_size 64 --sample_length 1 --buffer_size 100 \
#--exp_name $EXP$t --learning_rate 0.00001 --tensorboard $HOME_DIR"/tensorboardaudiovideo/" \
#--checkpoint_dir $HOME_DIR"/checkpointsaudiovideo/" --model UNet --datatype old --num_epochs 300 \
#--num_class 14 --block_size 12 --embedding 1 --mfcc 1 ---visual_init_checkpoint $HOME_DIR"/checkpointsaudiovideo/resnet/resnet_v1_50.ckpt" &
#
#        #echo "gpu: $i" &
#        P[$i]=$!
#        #echo "pid: ${P[$i]}"
#        if [ $i -eq 1 ]
#        then
#            wait ${P[0]}
#            wait ${P[1]}
#        fi
#        (( i = (i+1) % 2 ))
#done
