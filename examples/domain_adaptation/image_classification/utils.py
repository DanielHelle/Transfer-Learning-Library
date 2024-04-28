"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os
import os.path as osp
import time
from PIL import Image
import pickle
import numpy as np
import random
import numpy as np


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from tllib.utils.analysis import collect_feature, tsne, a_distance
from torch.utils.data import Dataset, ConcatDataset,Subset
from torch.utils.data import DataLoader

import tllib.vision.datasets as datasets
import tllib.vision.models as models
import custom_model
#import tllib.vision.models.cnn as cnn
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix, binary_accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

from torch.utils.data import TensorDataset
from torch.optim import SGD

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter

def stratified_cluster_sampling(features, labels, target_per_class_counts, k, n_clusters=10):
    # Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=k).fit(features)
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Organize data by clusters
    clusters = defaultdict(list)
    for idx, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(idx)

    selected_indices = []
    # Process each cluster
    for cluster_id, indices in clusters.items():
        # Determine the majority label in the cluster
        cluster_labels = labels[indices]
        label_counts = Counter(cluster_labels)
        majority_label, _ = label_counts.most_common(1)[0]

        # Select samples corresponding to the majority label that are closest to the centroid
        majority_indices = [idx for idx in indices if labels[idx] == majority_label]
        if not majority_indices:
            continue

        cluster_features = features[majority_indices]
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        sorted_indices = np.argsort(distances)

        count_needed = target_per_class_counts.get(majority_label, 0)
        selected_for_cluster = [majority_indices[i] for i in sorted_indices[:count_needed]]
        selected_indices.extend(selected_for_cluster)

    return selected_indices

def extract_features(data_loader, feature_extractor, device):
    features = []
    labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            extracted_features = feature_extractor(inputs).cpu().numpy()
            features.append(extracted_features)
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def a_distance_oversampling(source_loader, target_loader, feature_extractor, device, args, iterations):
    source_dataset = source_loader.dataset
    target_dataset = target_loader.dataset
    
    source_loader_for_features = DataLoader(source_dataset, batch_size=2, shuffle=False)
    target_loader_for_features = DataLoader(target_dataset, batch_size=32, shuffle=False)
    
    source_features, source_labels = extract_features(source_loader_for_features, feature_extractor, device)
    target_features, target_labels = extract_features(target_loader_for_features, feature_extractor, device)

    # Calculate per class counts from source dataset as reference
    unique_labels, counts = np.unique(source_labels, return_counts=True)
    source_class_count = dict(zip(unique_labels, counts))
    a_distances = []
    for k in range(iterations):
        if args.dataset_condensation == "True":
            # Apply stratified clustering only to target dataset
            target_indices = stratified_cluster_sampling(target_features, target_labels, source_class_count,k)
            adapted_target_dataset = Subset(target_dataset, target_indices)
            adapted_source_dataset = source_dataset
        else:
            # Apply stratified clustering to both datasets
            source_indices = stratified_cluster_sampling(source_features, source_labels, source_class_count,k)
            target_indices = stratified_cluster_sampling(target_features, target_labels, source_class_count,k)
            adapted_source_dataset = Subset(source_dataset, source_indices)
            adapted_target_dataset = Subset(target_dataset, target_indices)

        adapted_source_loader = DataLoader(adapted_source_dataset, batch_size=2, shuffle=True)
        adapted_target_loader = DataLoader(adapted_target_dataset, batch_size=2, shuffle=True)
        
        source_feature = collect_feature(adapted_source_loader, feature_extractor, device)
        target_feature = collect_feature(adapted_target_loader, feature_extractor, device)
        
        a_dist = calculate_a_distance(source_feature=source_feature, target_feature=target_feature,k=k, device=device, training_epochs=10)
        a_distances.append(a_dist)
    
    return sum(a_distances)/len(a_distances)

sys.path.append('../../..')

class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x
    
def calculate_a_distance(source_feature: torch.Tensor, target_feature: torch.Tensor, device,k, training_epochs=10, progress=True):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance

    Args:
        source_feature (torch.Tensor): Features from source domain.
        target_feature (torch.Tensor): Features from target domain.
        device (torch.device): Device to run the calculations on (CPU/GPU).
        training_epochs (int): Number of epochs to train the classifier.
        progress (bool): Whether to display progress.

    Returns:
        float: Calculated A-distance.
    """
    # Combine features and create labels
    source_label = torch.ones(source_feature.shape[0], 1)
    target_label = torch.zeros(target_feature.shape[0], 1)
    features = torch.cat([source_feature, target_feature], dim=0)
    labels = torch.cat([source_label, target_label], dim=0)

    # Stratified split into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=k
    )

    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=max(1, len(train_dataset) // 10), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, len(val_dataset) // 10), shuffle=False)

    anet = ANet(features.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        source_samples = 0
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
                    
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance

    
#A-distance without stratified sampling 

'''
def calculate_a_distance(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        source_samples = 0
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
                # Calculate validation loss for source samples
                if label.sum().item() > 0:  # If there are source samples in the batch
                    source_loss = F.binary_cross_entropy(y[label == 1], label[label == 1])
                    source_val_loss += source_loss.item() * label[label == 1].shape[0]
                    source_samples += label[label == 1].shape[0]
                    
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance

'''

#train_source_feature,val_source_feature, train_target_feature,val_target_feature,


def calculate(train_source_feature: torch.Tensor, val_source_feature: torch.Tensor,
              train_target_feature: torch.Tensor, val_target_feature: torch.Tensor,
              device, progress=True, training_epochs=10, patience=15):

    # Concatenate training and validation features
    train_feature = torch.cat([train_source_feature, train_target_feature], dim=0)
    val_feature = torch.cat([val_source_feature, val_target_feature], dim=0)
    
    # Create labels
    train_source_label = torch.ones((train_source_feature.shape[0], 1))
    train_target_label = torch.zeros((train_target_feature.shape[0], 1))
    
    val_source_label = torch.ones((val_source_feature.shape[0], 1))
    val_target_label = torch.zeros((val_target_feature.shape[0], 1))
    
    # Concatenate training and validation labels
    train_label = torch.cat([train_source_label, train_target_label], dim=0)
    val_label = torch.cat([val_source_label, val_target_label], dim=0)
    
    # Create datasets
    train_set = TensorDataset(train_feature, train_label)
    val_set = TensorDataset(val_feature, val_label)
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(train_feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        source_val_loss = 0.0
        source_samples = 0
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
                
                # Calculate validation loss for source samples
                if label.sum().item() > 0:  # If there are source samples in the batch
                    source_loss = F.binary_cross_entropy(y[label == 1], label[label == 1])
                    source_val_loss += source_loss.item() * label[label == 1].shape[0]

                    source_samples += label[label == 1].shape[0]

        source_val_loss /= source_samples

        

        # Check for overfitting on source samples
        if source_val_loss < best_val_loss:
            best_val_loss = source_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        

        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)

        if progress:
            print("epoch {} accuracy: {} A-dist: {} Source Val Loss: {}".format(epoch, meter.avg, a_distance, source_val_loss))

        if patience_counter >= patience:
            print("Early stopping due to overfitting on source samples.")
            break

    return a_distance



'''
def a_distance_oversampling(source_loader, target_loader, feature_extractor, device,args):
    source_dataset = source_loader.dataset
    target_dataset = target_loader.dataset
    oversample_train_factor = len(target_dataset) // len(source_dataset)
   
    if args.dataset_condensation == "True":
        oversampled_source_dataset = ConcatDataset([source_dataset] * oversample_train_factor)
        oversampled_source_dataloader = DataLoader(oversampled_source_dataset, batch_size=32, shuffle=True)
        target_shuffled_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

        source_feature = collect_feature(oversampled_source_dataloader, feature_extractor, device)
        target_feature = collect_feature(target_shuffled_loader, feature_extractor, device)
        a_dist = a_distance.calculate(source_feature=source_feature,target_feature=target_feature,device=device,training_epochs=10)

    else:
        source_dataloader = DataLoader(oversampled_source_dataset, batch_size=32, shuffle=True)
        target_shuffled_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
        source_feature = collect_feature(source_dataloader, feature_extractor, device)
        target_feature = collect_feature(target_shuffled_loader, feature_extractor, device)
        a_dist = a_distance.calculate(source_feature=source_feature,target_feature=target_feature,device=device,training_epochs=10)
    return a_dist
'''




def compute_average_a_distance(source_loader, target_loader, feature_extractor, device,args, k=5):

    num_source_images = len(source_loader.dataset)
    num_target_images = len(target_loader.dataset)
    
    fold_size_source = num_source_images // k
    fold_size_target = num_target_images // k
    
    a_distances = []

    for fold in range(k):
        
        # Define the start and end indices for the source and target datasets
        start_source = fold * fold_size_source
        end_source = (fold + 1) * fold_size_source
        start_target = fold * fold_size_target
        end_target = (fold + 1) * fold_size_target

        # Split datasets into train and validation for this fold
        train_source_subset = Subset(source_loader.dataset, list(range(0, start_source)) + list(range(end_source, num_source_images)))
        val_source_subset = Subset(source_loader.dataset, list(range(start_source, end_source)))
        
        train_target_subset = Subset(target_loader.dataset, list(range(0, start_target)) + list(range(end_target, num_target_images)))
        val_target_subset = Subset(target_loader.dataset, list(range(start_target, end_target)))
        
        # Oversampling source training subset
        oversample_train_factor = len(train_target_subset) // len(train_source_subset)
        oversampled_train_source = ConcatDataset([train_source_subset] * oversample_train_factor)
        
        # Oversampling source validation subset
        oversample_val_factor = len(val_target_subset) // len(val_source_subset)
        oversampled_val_source = ConcatDataset([val_source_subset] * oversample_val_factor)

        if args.dataset_condensation == "True":
            train_source_dataloader = DataLoader(oversampled_train_source, batch_size=32, shuffle=True)
            val_source_dataloader = DataLoader(oversampled_val_source, batch_size=32, shuffle=True)
            train_target_dataloader = DataLoader(train_target_subset, batch_size=32, shuffle=True)
            val_target_dataloader = DataLoader(val_target_subset, batch_size=32, shuffle=True)

        if args.dataset_condensation == "False":
            train_source_dataloader = DataLoader(train_source_subset, batch_size=32, shuffle=True)
            val_source_dataloader = DataLoader(val_source_subset, batch_size=32, shuffle=True)
            train_target_dataloader = DataLoader(train_target_subset, batch_size=32, shuffle=True)
            val_target_dataloader = DataLoader(val_target_subset, batch_size=32, shuffle=True)

        # Extract features
        train_source_feature = collect_feature(train_source_dataloader, feature_extractor, device)
        val_source_feature = collect_feature(val_source_dataloader, feature_extractor, device)
        train_target_feature = collect_feature(train_target_dataloader, feature_extractor, device)
        val_target_feature = collect_feature(val_target_dataloader, feature_extractor, device)

        # Calculate A-distance based on the training features
        A_distance = calculate(train_source_feature,val_source_feature, train_target_feature,val_target_feature, device, True)
        a_distances.append(A_distance.cpu())

       

    # Calculate average and standard deviation of A-distance over all training partitions
    print("A-distances: ", a_distances)
    avg_a_distance = np.mean(a_distances)
    std_dev = np.std(a_distances)
    
    return avg_a_distance, std_dev

class CondensedData(Dataset):
  #dataset_list is a list with two tensors, dataset_list[0]
  def __init__(self, dataset_list):
    self.dataset_list = dataset_list
      
    #count labels
    self.len = len(dataset_list[1])
    

  def __getitem__(self, index):
   
    img = self.dataset_list[0][index]
    #print(img)
    label = self.dataset_list[1][index]
    #print(label)


    return img, label

  def __len__(self):
    return self.len   


def get_model_names():
  
    return sorted(
        name for name in models.__dict__.keys()
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models() + custom_model.models

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def compare_backbone_and_loaded_weights(backbone,weights):
    print("Comparing shapes of ConvNet weights and loaded weights:")
    for name, param in backbone.named_parameters():
        if name in weights:
            loaded_param_shape = weights[name].shape
            conv_net_param_shape = param.shape

            if loaded_param_shape != conv_net_param_shape:
                print(f"Shape mismatch: {name}")
                print(f"  ConvNet shape: {conv_net_param_shape}")
                print(f"  Loaded shape: {loaded_param_shape}")
            else:
                print(f"Shapes match for {name}")
        else:
            print(f"{name} not found in loaded weights")

def get_model(model_name, pretrain=True,channel=3,num_classes=10,args = None):
    if args.source[0].lower() == "usps" or args.source[0].lower() == "mnist":
        channel = 1
    
    print(model_name)
    if model_name =="convnet":
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        backbone = custom_model.convnet(pretrained=False,channel=channel,num_classes=num_classes,net_width=net_width,net_depth=net_depth,net_act=net_act,net_norm=net_norm,net_pooling=net_pooling)
       
        #Load weights from dataset condensation to backbone
        #change this so it adapts to the scratch flag
        if args.convnet_weights_data_path != "none":
            weights = torch.load(args.convnet_weights_data_path)
            with torch.no_grad():
                #compare_backbone_and_loaded_weights(backbone,weights)
                #backbone.classifier.weight = nn.Parameter(weights['classifier.weight'])
                #backbone.classifier.bias = nn.Parameter(weights['classifier.bias'])
                updated_weights_dict = {k: v for k, v in weights.items() if not k.startswith('classifier')}
                backbone.load_state_dict(updated_weights_dict,strict=False)
                

    elif model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']



def save_tuple_as_numpy(data,i,dataset_name,current_dir):
    tensor_path =  osp.join(current_dir,'data','pre_cond',dataset_name.lower(),'imgs','train_source'+str(i)+".npy")
    label_path = osp.join(current_dir,'data','pre_cond',dataset_name.lower(),'labels','train_source_label'+str(i)+".npy")
    #tensor, label,int_value = data
    #array = tensor.numpy()
    if len(data) == 3:
        array, label,int_value = data
    elif len(data) == 2:
        array,label = data
    np.save(tensor_path, array.numpy(),allow_pickle=True)
    np.save(label_path, label,allow_pickle=True)

#helper function used for downloading pre-transformed datasets
def download_dataset(args):
    train_transform = get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip= args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std,no_aug=args.no_aug)
    val_transform = get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    
    #tensor_train_source = torch.tensor(train_source_dataset.data)
    current_dir = os.getcwd()

    if not os.path.exists(osp.join(current_dir,'data','pre_cond')):
 
        os.makedirs(osp.join(current_dir,'data','pre_cond'))

    if not os.path.exists(osp.join(current_dir,'data','pre_cond',str(args.data).lower())):
    
        os.makedirs(osp.join(osp.join(current_dir,'data','pre_cond',str(args.data).lower())))
    

    train_source_path = osp.join(current_dir,'data','pre_cond',str(args.data),'train_source.pkl')
    num_classes_path = osp.join(current_dir,'data','pre_cond',str(args.data),'num_classes.pkl')
    class_names_path = osp.join(current_dir,'data','pre_cond',str(args.data),'class_names_.pkl')
    data_path = osp.join(current_dir,'data','pre_cond',str(args.data).lower(),'imgs')
    label_path = osp.join(current_dir,'data','pre_cond',str(args.data).lower(),'labels')

    if not os.path.exists(data_path):
   
        os.makedirs(data_path)

    if not os.path.exists(label_path):
   
        os.makedirs(label_path)
   
    
    #If folder already populated remove tensors
    if len(os.listdir(data_path)) != 0:
        for file_name in os.listdir(data_path):
            try:
                os.remove(osp.join(data_path,file_name))
            except Exception as e:
                print("failed to delete {}. Reason: {}".format(osp.join(data_path,file_name), e))
        for file_name in os.listdir(label_path):
            try:
                os.remove(osp.join(label_path,file_name))
            except Exception as e:
                print("failed to delete {}. Reason: {}".format(osp.join(label_path,file_name), e))

    for i,data in enumerate(train_source_dataset):
         save_tuple_as_numpy(data,i,str(args.data),current_dir)
         

             
        
    #torch.save(num_classes, osp.join(current_dir,'data','pre_cond',str(args.data).lower(),"num_classes"))
    #torch.save(args.class_names, osp.join(current_dir,'data','pre_cond',str(args.data).lower(),"class_names"))

def get_condensed_source(dataset_name,source, args):
    abs_file_path = args.condensed_data_path
    condensed_data = torch.load(abs_file_path)
    return CondensedData(condensed_data["data"][0])
   


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names

def validate_5_fold(test_loader,model,args,device) -> float:

    dataset = test_loader.dataset

    # Print the original dataset size
    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")

    # Discard a sample if necessary before shuffling
    if original_size % 5 != 0:
        discard_samples = original_size % 5
        new_size = original_size - discard_samples
    else:
        discard_samples = 0
        new_size = original_size

    # Print the new dataset size after discarding
    print(f"Dataset size after discarding {discard_samples} sample(s): {new_size}")

    partition_size = new_size // 5

    # Ensure reproducibility
    np.random.seed(0)
    
    # Randomly shuffle the dataset indices
    indices = np.random.permutation(new_size)

    accuracies = []

    for i in range(5):
        # Determine the indices for the validation set
        val_indices = indices[i * partition_size:(i + 1) * partition_size]
        
        # Create the validation set
        val_set = torch.utils.data.Subset(dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_loader.batch_size, shuffle=False)
        
        # Validate the model
        acc1 = validate(val_loader, model, args, device)
        accuracies.append(acc1)
        print(f"test_acc1_partition{i + 1}: {acc1}")

    # Calculate and print the mean and standard deviation
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    

    return mean_acc, std_acc






def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None,no_aug = "False"):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
        
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
       
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
        
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if no_aug == "True":
        T.Compose(transforms)
        

    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
        
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
       
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
           
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
          
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    if no_aug == "True": 
        transforms.extend([
            T.ToTensor()
        ])
    else:
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    
    #if args.target[0].lower() == "mnist" or args.target[0].lower() == "usps":
    #    return T.Compose([
    #    transform,
    #    T.ToTensor(),
    #    T.Normalize(mean=(0.485,0.456), std=(0.229,0.224))
    #])

    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
