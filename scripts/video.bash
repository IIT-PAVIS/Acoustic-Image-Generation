#!/bin/bash
INITIAL_PATH=/data/sounddataset/short2
cd $INITIAL_PATH
for CLASS in $(ls -1 | grep class_ | sort)
do
	cd $CLASS
		for ELEM in $(ls -1 | grep data_ | sort)
		do
			echo  $INITIAL_PATH/$CLASS/$ELEM
			CUDA_VISIBLE_DEVICES=1 python3 /home/vsanguineti/Documents/Code/encoder-decoder/showvideo.py --model UNet --train_file \
      $INITIAL_PATH/$CLASS/$ELEM/testing_file.txt --batch_size 4 --data_type 'outdoor' \
      --init_checkpoint /data/checkpointsaudiovideo/unet1skip/Unetacresnet1conn_outdoor_1/epoch_17.ckpt
		done
		cd ..
done