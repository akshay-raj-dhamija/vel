### Experiments for Feature Distribution Techniques

## ODIN
```
python ODIN.py --arch resnet50 --output_dir /tmp/adhamija/ODIN --temperature 1 2 5 10 --images-path /scratch/datasets/ImageNet/ILSVRC_2012/val_in_folders/
```

Prerequsite --> Pre-extracted features

## OpenMax
_cosine distance on fc_
```
python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.01_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.1_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.2_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.3_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.4_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.5_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.6_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.7_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.8_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.9_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_1.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_5.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_10.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.01_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.1_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.2_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.3_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.4_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.5_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.6_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.7_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.8_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_0.9_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_1.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_5.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/OpenMax/fc/OpenMax/TS_10.0_DM_1.00/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/fc/OpenMax/cosine/
```

_euclidean distance on fc_
```
python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.01_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.1_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.2_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.3_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.4_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.5_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.6_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.7_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.8_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.9_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_1.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_5.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_10.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.01_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.1_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.2_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.3_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.4_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.5_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.6_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.7_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.8_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_0.9_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_1.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_5.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/OpenMax/fc/OpenMax/TS_10.0_DM_1.00/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/fc/OpenMax/euclidean/
```


_cosine distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.01_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.1_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.2_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.3_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.4_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.5_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.6_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.7_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.8_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.9_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_1.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_5.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_10.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.01_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.1_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.2_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.3_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.4_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.5_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.6_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.7_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.8_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_0.9_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_1.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_5.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/OpenMax/avgpool/OpenMax/TS_10.0_DM_1.00/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/avgpool/OpenMax/cosine/
```

_euclidean distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo OpenMax \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.01_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.1_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.2_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.3_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.4_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.5_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.6_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.7_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.8_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.9_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_1.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_5.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_10.0_DM_1.00/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.01_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.1_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.2_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.3_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.4_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.5_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.6_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.7_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.8_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_0.9_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_1.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_5.0_DM_1.00/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/OpenMax/avgpool/OpenMax/TS_10.0_DM_1.00/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/avgpool/OpenMax/euclidean/
```

## EVM
_cosine distance on fc_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10000 30000 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.01_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.1_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.2_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.3_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.4_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.5_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.6_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.7_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.8_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.9_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_1.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_5.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_10.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_10000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_30000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.01_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.1_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.2_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.3_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.4_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.5_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.6_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.7_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.8_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_0.9_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_1.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_5.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_10.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_10000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/fc/EVM/TS_30000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/fc/EVM/cosine/
```

[comment]: <> ( "TS 10000" "TS 30000")
_euclidean distance on fc_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10000 30000 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.01_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.1_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.2_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.3_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.4_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.5_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.6_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.7_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.8_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.9_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_1.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_5.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_10.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_10000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_30000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.01_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.1_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.2_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.3_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.4_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.5_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.6_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.7_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.8_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_0.9_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_1.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_5.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_10.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_10000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/fc/EVM/TS_30000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/fc/EVM/euclidean/
```

_cosine distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10000 30000 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/ --chunk_size 100
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.01_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.1_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.2_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.3_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.4_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.5_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.6_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.7_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.8_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.9_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_1.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_5.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_10.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_10000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_30000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.01_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.1_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.2_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.3_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.4_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.5_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.6_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.7_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.8_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_0.9_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_1.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_5.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_10.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_10000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/avgpool/EVM/TS_30000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/avgpool/EVM/cosine/
```

_euclidean distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 5 10 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10000 30000 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/ --chunk_size 100
```

_Evaluation_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.01_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.1_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.2_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.3_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.4_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.5_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.6_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.7_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.8_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.9_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_1.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_5.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_10.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_10000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_30000.0_DM_0.55_CT_0.70/ILSVRC_2012/val_in_folders.hdf5   \
		--unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.01_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.1_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.2_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.3_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.4_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.5_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.6_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.7_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.8_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_0.9_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_1.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_5.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_10.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_10000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
				/net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/avgpool/EVM/TS_30000.0_DM_0.55_CT_0.70/360_openset/val_in_folders.hdf5   \
		--all_layer_names probs probs probs probs probs probs probs probs probs probs probs probs probs   \
		--use-softmax-normalization True True True True True True True True True True True True True  \
		--label_names 'TS 0.01' 'TS 0.1' 'TS 0.2' 'TS 0.3' 'TS 0.4' 'TS 0.5' 'TS 0.6' 'TS 0.7' 'TS 0.8' 'TS 0.9' 'TS 1.0' 'TS 5.0' 'TS 10.0'  \
		--output_dir Thesis/Feature_Distribution/FD/avgpool/EVM/euclidean/
```




### Large tail size EVM's
_euclidean distance on fc_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 50000 75000 90000 100000 150000 175000 200000 250000 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/euclidean/EVM/ \
    --chunk_size 100
```

_cosine distance on fc_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names fc \
    --tailsize 50000 75000 90000 100000 150000 175000 200000 250000 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/fc/resnet50/cosine/EVM/ \
    --chunk_size 100
```



_euclidean distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 50000 75000 90000 100000 150000 175000 200000 250000 \
    --distance_metric euclidean \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/euclidean/EVM/ \
    --chunk_size 25
```

_cosine distance on avgpool_
```
python Feature_Distribution.py --OOD_Algo EVM \
    --training_knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/train.hdf5 \
    --testing_files /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                    /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/360_openset/val_in_folders.hdf5 \
    --layer_names avgpool \
    --tailsize 50000 75000 90000 100000 150000 175000 200000 250000 \
    --distance_metric cosine \
    --output_dir /net/reddwarf/bigscratch/adhamija/The/Features/default_resnet50_for_FD/avgpool/resnet50/cosine/EVM/ \
    --chunk_size 25
```
