#MNIST --> USPS
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_M2U/0/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 1 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_M2U/1/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 2 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_M2U/2/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 3 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_M2U/3/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 32 --seed 4 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_M2U/4/



#USPS --> MNSIT
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_U2M/0/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 1 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_U2M/1/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 2 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_U2M/2/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 3 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_U2M/3/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 32 --seed 4 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_U2M/4/


#SVHN --> MNISTRGB
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2M/0/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 1 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2M/1/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 2 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2M/2/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 3 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2M/3/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 32 --seed 4 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2M/4/




#SVHN --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 0 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2U/0/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 1 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2U/1/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 2 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2U/2/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 3 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2U/3/

CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 32 --seed 4 --epochs 60 --partition-source 0.8 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment_baseline_new/Digits_S2U/4/
