#export TMPDIR="/net/reddwarf/bigscratch/adhamija/The/FR/moco_v2_800ep_pretrain/tsp_logs/"
#export TMPDIR="/scratch/adhamija/tsp_logs/"
#outputpath="/net/reddwarf/bigscratch/adhamija/The/FR/moco_v2_800ep_pretrain/"
outputpath="/tmp/adhamija"
# CenterLoss --schedule 60 90 120 150 180
learning_rates=(10.0 1.0 1e-1 1e-2 1e-3)
second_loss_weights=(5.0 2.0 1.0 1e-1 1e-2 1e-3)
for lr in "${learning_rates[@]}"; do
  for second_loss_weight in "${second_loss_weights[@]}"; do
    tsp sh -c "python moco_lincls.py --data /scratch/datasets/ImageNet/ILSVRC_2012/ --epochs 200 \
              --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
              --pretrained /home/adhamija/moco/moco_v2_800ep_pretrain.pth.tar \
              --output-path $outputpath \
              --objecto_layer 100 --approach CenterLoss  --schedule 60 90 120 150 180 \
              --lr $lr --second_loss_weight $second_loss_weight"
  done
done
# CenterLoss with cosine lr scheduling
learning_rates=(10.0 1.0 1e-1 1e-2 1e-3)
second_loss_weights=(5.0 2.0 1.0 1e-1 1e-2 1e-3)
for lr in "${learning_rates[@]}"; do
  for second_loss_weight in "${second_loss_weights[@]}"; do
    tsp sh -c "python moco_lincls.py --data /scratch/datasets/ImageNet/ILSVRC_2012/ --epochs 200 \
              --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
              --pretrained /home/adhamija/moco/moco_v2_800ep_pretrain.pth.tar \
              --output-path $outputpath \
              --objecto_layer 100 --approach CenterLoss --cos \
              --lr $lr --second_loss_weight $second_loss_weight"
  done
done