R#!/usr/bin/env bash
HOME_DIR="/data/"
ROOT_DIR="/home/vsanguineti/Datasets/dualcam_actions_dataset/1_second/lists/"
EXP=/data/checkpointsaudiovideo/unet1skipoldmoreepochs/Unetacresnet1conn_actionsmoreepochs_
NUM=(1 2 3 4 5)

cd ..

for t in "${NUM[@]}"
do
MODEL_PATH=$EXP$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
echo $EPOCH_FILE
  CUDA_VISIBLE_DEVICES=0 python3 extract_features_unetraces.py --model UNet --train_file $ROOT_DIR"/testing.txt" --batch_size 16 \
  --datatype old --init_checkpoint $EPOCH_FILE --nr_frames 12 --encoder_type Audio &

  CUDA_VISIBLE_DEVICES=1 python3 extract_features_unetraces.py --model UNet --train_file $ROOT_DIR"/training.txt" --batch_size 16 \
  --datatype old --init_checkpoint $EPOCH_FILE --nr_frames 12 --encoder_type Audio
done

NUM=(1 2 3 4 5)
for t in "${NUM[@]}"
do
MODEL_PATH=$EXP$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
echo $EPOCH_FILE
  CUDA_VISIBLE_DEVICES=0 python3 knn.py $EPOCH_FILE Audio testing
done