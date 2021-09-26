# OpenMax cosine FC
tsp sh -c "python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/resnet50/cosine/"

# OpenMax euclidiean FC
tsp sh -c "python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/resnet50/euclidean/"

# OpenMax cosine avgpool
tsp sh -c "python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/resnet50/cosine/"

# OpenMax euclidiean avgpool
tsp sh -c "python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/resnet50/euclidean/"
