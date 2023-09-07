#MNIST --> USPS
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment_baseline/Digits_M2U/

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment_baseline/Digits_U2M/

#SVHN 1ipc --> MNISTRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment_baseline/Digits_S2M/

#SVHN 1ipc --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment_baseline/Digits_S2U/


