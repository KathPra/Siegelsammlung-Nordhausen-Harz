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
outname= "fine_tune_batchsize10_epoch50_NR" # is also model name

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

## Load data - adjust to model
input_size = 299                                       # required for inception net

data_transforms = transforms.Compose([
    transforms.Resize((input_size,input_size)),                    
    #transforms.CenterCrop(224),                 
    transforms.ToTensor()#,                    
#     transforms.Normalize(                      
#     mean = [0.485, 0.456, 0.406],
#     std = [0.229, 0.224, 0.225]                  
# )
])

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