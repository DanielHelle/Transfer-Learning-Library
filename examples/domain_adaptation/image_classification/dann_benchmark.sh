#lr = 0.01 * sqrt(k) where k is what we multiply batch size 32 with, ie 1/2.
#CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --batch-size 16 --lr 0.0071 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --download-dataset-only True --seed 1 --log logs/dann/Office31_A2W