ROOT_DIR="/home/vsanguineti/Datasets/tfrecords2/lists/"
cd ..
EXP=/data/checkpoints/variationalfeatures
MOD2=(Ac AcTrue)

for ((j = 0 ; j < ${#EXP[@]} ; j++))
do
  BEST_EPOCH=$(grep "Epoch" ${EXP[$j]}"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
  EPOCH_FILE=${EXP[$j]}"/epoch_"$BEST_EPOCH".ckpt"

  CUDA_VISIBLE_DEVICES=1 python3 extract_j.py --model UNet --train_file $ROOT_DIR"/testing.txt" --batch_size 4 --num_classes 10 \
  --init_checkpoint $EPOCH_FILE --nr_frames 1 --onlyaudiovideo 1

  CUDA_VISIBLE_DEVICES=1 python3 extract_j.py --model UNet --train_file $ROOT_DIR"/training.txt" --batch_size 4 --num_classes 10 \
    --init_checkpoint $EPOCH_FILE --nr_frames 1 --onlyaudiovideo 1

  echo $EPOCH_FILE

  CUDA_VISIBLE_DEVICES=1 python3 retrieve.py $EPOCH_FILE AcTrue Ac testing outdoor

  for ((i = 0 ; i < ${#MOD2[@]} ; i++))
  do

    CUDA_VISIBLE_DEVICES=1 python3 knn.py $EPOCH_FILE ${MOD2[$i]} testing

  done


done
