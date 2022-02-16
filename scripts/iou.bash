cd ..

#for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#CUDA_VISIBLE_DEVICES=1 python3 iouenergythreshold.py --model UNet --train_file \
#/home/vsanguineti/Datasets/dualcam_actions_dataset/1_second/lists/testing.txt \
#--batch_size 16 --plot 0 --init_checkpoint /data/checkpointsaudiovideo/UnetResAcframesaction1secondinizializza/epoch_30.ckpt \
#--nr_frames 1 --threshold $i --datatype old
#done
#
#CUDA_VISIBLE_DEVICES=1 python3 areaundercurve.py --init_checkpoint \
#/data/checkpointsaudiovideo/UnetResAcframesaction1secondinizializza/epoch_30.ckpt

#for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#CUDA_VISIBLE_DEVICES=1 python3 iouenergythreshold.py --model UNet --train_file \
#/home/vsanguineti/Datasets/tfrecords2/lists/testing.txt  \
#--batch_size 16 --plot 0 --init_checkpoint /data/checkpointsaudiovideo/UNetoutdoorregl2/epoch_24.ckpt \
#--nr_frames 1 --threshold $i --datatype outdoor
#done
#
#CUDA_VISIBLE_DEVICES=1 python3 areaundercurve.py --init_checkpoint \
#/data/checkpointsaudiovideo/UNetoutdoorregl2/epoch_24.ckpt

#for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#CUDA_VISIBLE_DEVICES=0 python3 iouenergythreshold.py --model UNet --train_file \
#/home/vsanguineti/Datasets/acviwtfrecords/lists/testing.txt  \
#--batch_size 16 --plot 0 --init_checkpoint /data/checkpoints/UnetResAcframestwoconcconvloss6-f/epoch_15.ckpt \
#--nr_frames 1 --threshold $i --datatype outdoor
#done
#
#CUDA_VISIBLE_DEVICES=0 python3 areaundercurve.py --init_checkpoint \
#/data/checkpoints/UnetResAcframestwoconcconvloss6-f/epoch_15.ckpt

#iou
#for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#CUDA_VISIBLE_DEVICES=0 python3 showimages_bb.py --model UNet --train_file \
#/data/learningtolocalizesoundsource/test.txt  \
#--plot 0 --init_checkpoint /data/checkpointsaudiovideo/UNetoutdoorregl2/epoch_24.ckpt \
# --threshold $i
#done
#CUDA_VISIBLE_DEVICES=0 python3 areaundercurve.py --init_checkpoint \
#/data/checkpointsaudiovideo/UNetoutdoorregl2/epoch_24.ckpt --flickr 1

for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
CUDA_VISIBLE_DEVICES=0 python3 showimages_bb.py --model UNet --train_file \
/data/learningtolocalizesoundsource/test.txt  \
--plot 0 --init_checkpoint /data/checkpointsaudiovideo/UnetResAcframesaction1secondinizializza/epoch_30.ckpt \
 --threshold $i
done
CUDA_VISIBLE_DEVICES=0 python3 areaundercurve.py --init_checkpoint \
/data/checkpointsaudiovideo/UnetResAcframesaction1secondinizializza/epoch_30.ckpt --flickr 1

