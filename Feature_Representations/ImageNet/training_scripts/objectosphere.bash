outputpath="/net/reddwarf/bigscratch/adhamija/The/FR/moco_v2_800ep_pretrain/"
#outputpath="/tmp/adhamija"
tsp_logs_dir=$outputpath/"tsp_logs/objectosphere"

mkdir -p $tsp_logs_dir
export TMPDIR=$tsp_logs_dir
export TS_SOCKET=$tsp_logs_dir/$HOSTNAME.tsp_socket

# CenterLoss with cosine lr scheduling
learning_rates=(1.0 1e-1 1e-2)
second_loss_weights=(5.0 2.0 1.0 1e-1)
for lr in "${learning_rates[@]}"; do
  for second_loss_weight in "${second_loss_weights[@]}"; do
    tsp sh -c "python moco_lincls.py --data /scratch/datasets/ImageNet/ILSVRC_2012/ --epochs 200 \
              --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
              --pretrained /home/adhamija/moco/moco_v2_800ep_pretrain.pth.tar \
              --output-path $outputpath \
              --objecto_layer 100 --approach objectosphere --cos \
              --lr $lr --second_loss_weight $second_loss_weight"
  done
done