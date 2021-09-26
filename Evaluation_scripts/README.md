### Experiments for evaluating network architectures

_ResNets_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet18/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet34/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet50/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet101/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet152/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet18/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet34/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet50/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet101/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/resnet152/360_openset/val_in_folders.hdf5 \
                --all_layer_names fc fc fc fc fc \
                --use-softmax-normalization True True True True True \
                --label_names ResNet-18 ResNet-34 ResNet-50 ResNet-101 ResNet-152 \
                --output_dir Thesis/NetworkArchitecturePlots/ResNets
```

_Dense Nets_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet121/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet161/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet169/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet201/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet121/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet161/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet169/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/densenet201/360_openset/val_in_folders.hdf5 \
                --all_layer_names classifier classifier classifier classifier \
                --use-softmax-normalization True True True True \
                --label_names DenseNet-121 DenseNet-161 DenseNet-169 DenseNet-201 \
                --output_dir Thesis/NetworkArchitecturePlots/DenseNets
```


_VGG Nets_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg11/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg13/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg16/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg19/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg11/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg13/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg16/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg19/360_openset/val_in_folders.hdf5 \
                --all_layer_names classifier:s:6 classifier:s:6 classifier:s:6 classifier:s:6 \
                --use-softmax-normalization True True True True \
                --label_names VGG-11 VGG-13 VGG-16 VGG-19 \
                --output_dir Thesis/NetworkArchitecturePlots/VGG
```


_VGG Nets with Batch Normalization_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg11_bn/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg13_bn/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg16_bn/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg19_bn/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg11_bn/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg13_bn/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg16_bn/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/vgg19_bn/360_openset/val_in_folders.hdf5 \
                --all_layer_names classifier:s:6 classifier:s:6 classifier:s:6 classifier:s:6 \
                --use-softmax-normalization True True True True \
                --label_names "VGG-11 (Batch Normalized)" "VGG-13 (Batch Normalized)" "VGG-16 (Batch Normalized)" "VGG-19 (Batch Normalized)" \
                --output_dir Thesis/NetworkArchitecturePlots/VGG_bn
```


_Low compute networks_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mnasnet0_5/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mnasnet1_0/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mobilenet_v2/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/shufflenet_v2_x0_5/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/shufflenet_v2_x1_0/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/squeezenet1_0/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/squeezenet1_1/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mnasnet0_5/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mnasnet1_0/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/mobilenet_v2/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/shufflenet_v2_x0_5/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/shufflenet_v2_x1_0/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/squeezenet1_0/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_pytorch_model.sh/squeezenet1_1/360_openset/val_in_folders.hdf5 \
                --all_layer_names classifier:s:1 classifier:s:1 classifier:s:1 fc fc classifier:s:3 classifier:s:3 \
                --use-softmax-normalization True True True True True True True \
                --label_names "MNASNet 0.5" "MNASNet 1.0" "MobileNet v2" "ShuffleNet v2 0.5x" "ShuffleNet v2 1.0x" "SqueezeNet 1.0" "SqueezeNet 1.1"\
                --output_dir Thesis/NetworkArchitecturePlots/Mobile
```

_DeiT_
```
python plot_creator.py --knowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_tiny_patch16_224/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_small_patch16_224/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_base_patch16_224/ILSVRC_2012/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_base_patch16_384/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_tiny_patch16_224/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_small_patch16_224/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_base_patch16_224/360_openset/val_in_folders.hdf5 \
                /net/reddwarf/bigscratch/adhamija/The/Features/extract_all_DeiT.sh/deit_base_patch16_384/360_openset/val_in_folders.hdf5 \
                --all_layer_names head head head head \
                --use-softmax-normalization True True True True \
                --label_names "Tiny patch16 224" "Small patch16 224" "Base patch16 224" "Base patch16 384" \
                --output_dir Thesis/NetworkArchitecturePlots/DeiT
```




_NFNet_
```
python plot_creator.py --knowns_files /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f0/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f1/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f2/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f3/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f4/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f5/ILSVRC_2012/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f6/ILSVRC_2012/val_in_folders.hdf5 \
             --unknowns_files /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f0/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f1/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f2/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f3/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f4/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f5/360_openset/val_in_folders.hdf5 \
                /net/patriot/scratch/adhamija/NFNet/dm_nfnet_f6/360_openset/val_in_folders.hdf5 \
                --all_layer_names head:s:fc head:s:fc head:s:fc head:s:fc head:s:fc head:s:fc head:s:fc \
                --use-softmax-normalization True True True True True True True \
                --label_names "NFNet F0" "NFNet F1" "NFNet F2" "NFNet F3" "NFNet F4" "NFNet F5" "NFNet F6"\
                --output_dir Thesis/NetworkArchitecturePlots/NFNet
```
