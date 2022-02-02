#outputpath="/tmp/adhamija"
outputpath="/net/reddwarf/bigscratch/adhamija/The/FR/moco_v2_800ep_pretrain/"
tsp_logs_dir=$outputpath/"tsp_logs/SoftMax"
#/scratch/adhamija/

mkdir -p $tsp_logs_dir
export TMPDIR=$tsp_logs_dir
export TS_SOCKET=$tsp_logs_dir/$HOSTNAME.tsp_socket

# SoftMax --schedule 60 90 120 150 180
learning_rates=(10.0 1.0 1e-1 1e-2 1e-3)
for lr in "${learning_rates[@]}"; do
  tsp sh -c "python moco_lincls.py --data /scratch/datasets/ImageNet/ILSVRC_2012/ --epochs 1 \
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
            --pretrained /home/adhamija/moco/moco_v2_800ep_pretrain.pth.tar \
            --output-path $outputpath \
            --objecto_layer 100 --approach SoftMax  --schedule 60 90 120 150 180 --lr $lr"
done
# SoftMax with cosine lr scheduling
learning_rates=(10.0 1.0 1e-1 1e-2 1e-3)
for lr in "${learning_rates[@]}"; do
  tsp sh -c "python moco_lincls.py --data /scratch/datasets/ImageNet/ILSVRC_2012/ --epochs 1 \
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
            --pretrained /home/adhamija/moco/moco_v2_800ep_pretrain.pth.tar \
            --output-path $outputpath \
            --objecto_layer 100 --approach SoftMax --cos --lr $lr"
done