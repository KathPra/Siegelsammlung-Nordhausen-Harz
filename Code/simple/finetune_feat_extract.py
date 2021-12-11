import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import os
import os.path
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import pickle5 as pickle
import numpy as np

data_dir = "/work-ceph/lprasse/siegel/data/siegel_gray_norm" 
batch_size= 200
device = torch.device("cuda:3")#"cuda:1" or "cpu"
outname= "fine_tune_batchsize10_epoch50_NR"
model_name = "inception"
# Number of classes in the dataset
num_classes = 4
# Batch size for training (change depending on how much memory you have)
batch_size = 5
# Number of epochs to train for
num_epochs = 200
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

## PREP
def save_as_pickle(obj, filename):
    """
    save an object in a pickle file dump
    :param obj: object to dump
    :param filename: target file
    :return:
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """
    load an object from a given pickle file
    :param filename: source file
    :return: loaded object
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                if is_inception and phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

## Load dataset and create data loader
image_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=25)
print("data loaded")

# Load filenames imagenet
filenames = []
for i in image_dataset.imgs:
  name = i[0]
  filenames.append(name)

print(len(filenames))   # 7790
save_as_pickle(filenames, f"/work-ceph/lprasse/siegel/features/{outname}/filenames.pkl")

## LOAD Model
model_ft = torch.load(f"/work/lprasse/Code/simple/Models/{outname}")   ## select which model to use ##
#print(model_ft)
## INCEPTION ONLY
model_ft.AuxLogits.fc = nn.Identity()
# Handle the primary net
model_ft.fc = nn.Identity()
#print(model_ft)
## ALL MODELS
model_ft = model_ft.to(device)
model_ft.eval()

print(model_ft)

print("start feature extraction")
#Generate features and save them

with torch.no_grad():
    i_batch=0
    for _, sample_batched in enumerate(dataloader):
      data = sample_batched[0]
      data = data.to(device)
      feature=model_ft(data)
      save_as_pickle(feature, f"/work-ceph/lprasse/siegel/features/{outname}/train_{i_batch}.pkl")
      print(i_batch)
      i_batch+=1