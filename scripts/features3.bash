ROOT_DIR="/home/vsanguineti/Datasets/tfrecords2/lists/"
cd ..
#/data/checkpoints/variationaltriplets /data/checkpoints/variationaltripletdifficile /data/checkpoints/variationalmoddropmean
# /data/checkpoints/variational6triplets /data/checkpoints/variationalmoddropmean /data/checkpoints/variationall2 /data/checkpoints/variationalncawithout /data/checkpoints/variationaltripleteasy
EXP=(/data/checkpoints/variationanoconctriplet )

MOD=(Ac Audio Video)
MOD2=(VideoAudio Audio Video)
for ((j = 0 ; j < ${#EXP[@]} ; j++))
do
  BEST_EPOCH=$(grep "Epoch" ${EXP[$j]}"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
  EPOCH_FILE=${EXP[$j]}"/epoch_"$BEST_EPOCH".ckpt"

  CUDA_VISIBLE_DEVICES=1 python3 extract_triplet.py --model UNet --train_file $ROOT_DIR"/training.txt" --batch_size 4 --num_classes 10 \
    --init_checkpoint $EPOCH_FILE --nr_frames 1

  CUDA_VISIBLE_DEVICES=1 python3 extract_triplet.py --model UNet --train_file $ROOT_DIR"/testing.txt" --batch_size 4 --num_classes 10 \
  --init_checkpoint $EPOCH_FILE --nr_frames 1

  CUDA_VISIBLE_DEVICES=1 python3 mean.py --model UNet --train_file $ROOT_DIR"/testing.txt" --batch_size 4 --num_classes 10 \
  --init_checkpoint $EPOCH_FILE --nr_frames 1
  #
  #CUDA_VISIBLE_DEVICES=1 python3 mean.py --model UNet --train_file $ROOT_DIR"/training.txt" --batch_size 4 --num_classes 10 \
  #--init_checkpoint $EPOCH_FILE --nr_frames 1

  echo $EPOCH_FILE

  for ((i = 0 ; i < ${#MOD[@]} ; i++))
  do

    CUDA_VISIBLE_DEVICES=1 python3 knn.py $EPOCH_FILE ${MOD[$i]} testing
    CUDA_VISIBLE_DEVICES=1 python3 retrieve.py $EPOCH_FILE ${MOD2[$i]} Ac testing outdoor

  done


done
