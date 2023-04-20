#for reproduction this value should be set: --seed 1

#Command for dann with office31, A2W
#CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a convnet -b 32 --epochs 20 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Office31_A2W --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/res_DC_pre_processed_office31_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/state_dict_DC_pre_processed_office31_ConvNet_10ipc.pt



#MNIST to USPS 1 channel
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/noise_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp5_ConvNet_ipc1_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_1ipc.pt







#Exp on utility DigitsM2U, baseline source input is random noise and no copied weights, seed 0
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/noise_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path none

#Exp on utility DigitsM2U, source input is random noise but weights are copied, seed 0
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/noise_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp5_ConvNet_ipc1_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_1ipc.pt

#Exp on utility DigitsM2U, source is 1ipc condensed MNIST, and weights are not copied
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp5_ConvNet_ipc1_no_aug_s_M_t_U_digits_80train_20test/res_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path none

#Exp on utility DigitsM2U, source is  1ipc condensed MNIST, weights are copied
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp5_ConvNet_ipc1_no_aug_s_M_t_U_digits_80train_20test/res_DC_pre_processed_mnist_ConvNet_1ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp5_ConvNet_ipc1_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_1ipc.pt

#Exp on utility DigitsM2U, source is 10ipc condensed MNIST, weights are copied
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 1 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/res_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_10ipc.pt

#Below got 60 accuracy, /home/daniel/exjobb/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/dann/Digits_M2U/utility_exp/train-2023-04-19-15_58_26.txt
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 2 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/res_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_10ipc.pt

#/home/daniel/exjobb/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/dann/Digits_M2U/utility_exp/train-2023-04-19-16_24_31.txt
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 2 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/noise_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path none

#/home/daniel/exjobb/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/dann/Digits_M2U/utility_exp/train-2023-04-19-16_43_32.txt
#CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 2 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/noise_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_10ipc.pt

#61.5% accuracy
CUDA_VISIBLE_DEVICES=0 python dann.py data/digits -d Digits -s MNIST -t USPS -a convnet -b 3 --seed 0 --epochs 100 --norm-mean 0.485 --norm-std 0.229 --resize-size 32 --train-resizing res. --val-resizing res. --log logs/dann/Digits_M2U/utility_exp --dataset-condensation True --condensed-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/res_DC_pre_processed_mnist_ConvNet_10ipc.pt --convnet-weights-data-path /home/daniel/exjobb/DatasetCondensation/result/exp6_ConvNet_ipc10_no_aug_s_M_t_U_digits_80train_20test/state_dict_DC_pre_processed_mnist_ConvNet_10ipc.pt