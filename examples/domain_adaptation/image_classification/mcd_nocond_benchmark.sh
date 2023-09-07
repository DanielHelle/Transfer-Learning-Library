#MNIST --> USPS
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --trade-off 0.3 --trade-off-entropy 0.03 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcd/experiment_baseline/Digits_M2U/

#USPS --> MNIST
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --trade-off 0.3 --trade-off-entropy 0.03 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcd/experiment_baseline/Digits_U2M/

#SVHN --> MNISTRGB
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --trade-off 0.3 --trade-off-entropy 0.03 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcd/experiment_baseline/Digits_S2M/

#SVHN  --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --trade-off 0.3 --trade-off-entropy 0.03 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcd/experiment_baseline/Digits_S2U/


