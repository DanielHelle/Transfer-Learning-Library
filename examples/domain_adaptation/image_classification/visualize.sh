#Visualization of USPS-->MNIST

#DANN

#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis


#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis

#ADDA
#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/adda/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis

#CDAN
#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python cdan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/cdan/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python cdan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/cdan/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis

#JAN
#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python jan.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/jan/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis

#MCC
#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python mcc.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcc/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python mcc.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/mcc/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis

#MCD

#USPS 1ipc --> MNIST
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 2 --seed 0 --epochs 40 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --trade-off 0.3 --trade-off-entropy 0.03 --train-resizing res. --val-resizing res. --log logs/mcd/experiment/Digits_U2M/1ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/res_DC_pre_processed_usps_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/1ipc/state_dict_DC_pre_processed_usps_ConvNet_1ipc.pt --phase analysis

#USPS 10ipc --> MNSIT
CUDA_VISIBLE_DEVICES=0 python mcd.py data/digits -d Digits -s USPS -t MNIST -a convnet -b 3 --seed 0 --epochs 60 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --trade-off 0.3 --trade-off-entropy 0.03 --train-resizing res. --val-resizing res. --log logs/mcd/experiment/Digits_U2M/10ipc/ --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/res_DC_pre_processed_usps_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/condensed_usps/10ipc/state_dict_DC_pre_processed_usps_ConvNet_10ipc.pt --phase analysis


'
