


#MNIST 1ipc --> USPS
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_M2U/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/1ipc/res_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/1ipc/state_dict_DC_pre_processed_mnist_ConvNet_1ipc.pt

#MNIST 10ipc --> USPS
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_M2U/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/10ipc/res_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/10ipc/state_dict_DC_pre_processed_mnist_ConvNet_10ipc.pt


#Below is finished
#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt

#SVHN 1ipc --> MNISTRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/res_DC_pre_processed_svhn_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/state_dict_DC_pre_processed_svhn_ConvNet_1ipc.pt

#SVHN 10ipc --> MNSITRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/res_DC_pre_processed_svhn_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/state_dict_DC_pre_processed_svhn_ConvNet_10ipc.pt

#SVHN 1ipc --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2U/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/res_DC_pre_processed_svhn_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/state_dict_DC_pre_processed_svhn_ConvNet_1ipc.pt

#SVHN 10ipc --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2U/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/res_DC_pre_processed_svhn_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/state_dict_DC_pre_processed_svhn_ConvNet_10ipc.pt


#Below is same as above but with randomly sampled source imgs as 1ipc and 10ipc
#MNIST 1ipc --> USPS Random source imgs and no copied weights
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_M2U/1ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/1ipc/real_init_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path none

#MNIST 10ipc --> USPS Random source imgs and no copied weights
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_M2U/10ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_mnist/10ipc/real_init_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path none

#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/1ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/real_init_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path none

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/10ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/real_init_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path none

#SVHN 1ipc --> MNISTRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2M/1ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/real_init_DC_pre_processed_svhn_ConvNet_1ipc.pt --convnet-weights-data-path none

#SVHN 10ipc --> MNSITRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2M/10ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/real_init_DC_pre_processed_svhn_ConvNet_10ipc.pt --convnet-weights-data-path none

#SVHN 1ipc --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2U/1ipc/real_init/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/1ipc/real_init_DC_pre_processed_svhn_ConvNet_1ipc.pt --convnet-weights-data-path none

#SVHN 10ipc --> USPSRGB
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t USPSRGB -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_S2U/10ipc/real_init --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_svhn/10ipc/real_init_DC_pre_processed_svhn_ConvNet_10ipc.pt --convnet-weights-data-path none