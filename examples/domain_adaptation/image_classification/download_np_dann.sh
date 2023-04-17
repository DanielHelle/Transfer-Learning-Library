#lr = 0.01 * sqrt(k) where k is what we multiply batch size 32 with, ie 1/2.
#CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --batch-size 16 --lr 0.0071 --epochs 20 --seed 1 --log logs/dann/Office31_A2W


#1. Command to download office source Amazon, no augmentation
#CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet18 --download-dataset-only True --no-aug True --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Office31_A2W


CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s M -t U -a convnet --download-dataset-only True --no-aug True --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U


#2. Command to download