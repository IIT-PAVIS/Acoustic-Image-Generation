#!/usr/bin/env bash
HOME_DIR="/data/"
ROOT_DIR="/home/vsanguineti/Datasets/tfrecords2/lists/"
EXP=/data/checkpointsaudiovideo/AE1skip/Unetacresnetnolatent1skip_outdoor_
NUM=(1 2 3 4 5)
#EXP2=/data/checkpointsaudiovideo/dualcamnet12/Dualcamnet_outdoor_
cd ..

for t in "${NUM[@]}"
do
MODEL_PATH=$EXP$t
BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"
echo $EPOCH_FILE

#CUDA_VISIBLE_DEVICES=0 python3 main.py --mode test --train_file $ROOT_DIR"/training.txt" --valid_file \
#$ROOT_DIR"/validation.txt" --test_file \
#$ROOT_DIR"/testing.txt" --batch_size 16 \
#--sample_length 1 --buffer_size 100 --exp_name $EXP$t --tensorboard \
#$HOME_DIR"/tensorboaraudiovideo/" --checkpoint_dir $HOME_DIR"/checkpointsaudiovideo/" --model UNet \
#--datatype old --num_class 14 --block_size 12 --mfcc 1 \
#--embedding 1 --restore_checkpoint $EPOCH_FILE

#for t2 in "${NUM[@]}"
#do
#
#MODEL_PATH2=$EXP2$t2
#BEST_EPOCH2=$(grep "Epoch" $MODEL_PATH2"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
#EPOCH_FILE2=$MODEL_PATH2"/epoch_"$BEST_EPOCH2".ckpt"
#echo $EPOCH_FILE2
#CUDA_VISIBLE_DEVICES=1 python3  saveimagesresnet.py --train_file $ROOT_DIR"/testing.txt"  \
#--batch_size 16 --datatype old --init_checkpoint $EPOCH_FILE \
#--ac_checkpoint $EPOCH_FILE2
#
#done
#
#CUDA_VISIBLE_DEVICES=1 python3  iouenergythreshold.py --model UNet --train_file $ROOT_DIR"/testing.txt"  \
#--batch_size 16 --init_checkpoint $EPOCH_FILE \
#--nr_frames 12 --datatype old
#
#CUDA_VISIBLE_DEVICES=1 python3 showimages_bb.py --model UNet --train_file $HOME_DIR"/learningtolocalizesoundsource/test.txt" \
#--datatype old --init_checkpoint $EPOCH_FILE \
#--threshold 0.5 --plot 0

for i in 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0
do
CUDA_VISIBLE_DEVICES=0 python3 showimages_bb.py --model UNet --train_file \
/data/learningtolocalizesoundsource/test.txt  --init_checkpoint $EPOCH_FILE \
 --threshold $i --plot 0 --batch_size 16 --ae 1
done
CUDA_VISIBLE_DEVICES=0 python3 areaundercurve.py --init_checkpoint \
$EPOCH_FILE --flickr 1

for i in 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0
do
CUDA_VISIBLE_DEVICES=0 python3 iouenergythreshold.py --model UNet --train_file \
$ROOT_DIR"/testing.txt"  \
--batch_size 16 --plot 0 --init_checkpoint $EPOCH_FILE \
--nr_frames 12 --threshold $i --datatype outdoor --ae 1
done

CUDA_VISIBLE_DEVICES=0 python3 areaundercurve.py --init_checkpoint $EPOCH_FILE

done