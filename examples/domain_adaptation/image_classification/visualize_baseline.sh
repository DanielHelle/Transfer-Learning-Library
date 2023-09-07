#Visualization U2M alg baselines for all algorithms. 

#DANN USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/experiment_baseline/Digits_U2M/ --phase analysis

#MCC USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python mcc.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcc/experiment_baseline/Digits_U2M/ --phase analysis


#CDAN USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python cdan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/cdan/experiment_baseline/Digits_U2M/ --phase analysis


#JAN USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline/Digits_U2M/ --phase analysis

#MCD USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --trade-off 0.3 --trade-off-entropy 0.03 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcd/experiment_baseline/Digits_U2M/ --phase analysis
