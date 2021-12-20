### USER INPUT: complete path to image folder
# Attention: path must exist already, it cannot be created by this code!
data_dir = "/work-ceph/lprasse/siegel/data/siegel_gray_norm"
outname= "pretrained"                # is also model name
out_path = "/work-ceph/lprasse/siegel/features/"

### NO USER INPUT REQUIRED
### Parameters that may be altered: batch_size (change according to memory availability), CUDA/GPU (change according to availability),
### model_name, num_workers, input_size (must match the model loaded), use_pretrained (True/False)

### Python packages used
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

### Model settings
batch_size= 200
device = torch.device("cuda:3")#"cuda:1" or "cpu"

### Functions defined
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

### Function calls
## Load data - adjust to model
input_size = 299                                       # required for inception net

data_transforms = transforms.Compose([
    transforms.Resize((input_size,input_size)),                                 
    transforms.ToTensor()
])

## Load dataset and create data loader
image_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=25)
print("data loaded")

# Retrieve filenames from loaded dataset
filenames = []
for i in image_dataset.imgs:
  name = i[0]
  filenames.append(name)
# Check if all filenames are loaded
print(len(filenames))   
# Save filenames for later retrieval during clustering
# Important: This only works as long as shaffle = False is specified in the dataloader
save_as_pickle(filenames, os.path.join(out_path, outname, "filenames.pkl"))


## Load model
model_ft = models.inception_v3(pretrained=True)  ## select which model to use ##
#print(model_ft) # display model with all parameters
## Remove the classification header to have the extracted features, not a prediction
model_ft.AuxLogits.fc = nn.Identity()   # auxilary net
model_ft.fc = nn.Identity()             # primary net

# Prepare model for evaluation
model_ft = model_ft.to(device)
model_ft.eval()

print("start feature extraction")
# Generate features and save them
with torch.no_grad():
    i_batch=0
    for _, sample_batched in enumerate(dataloader):
      data = sample_batched[0]
      data = data.to(device)
      feature=model_ft(data)
      outpath = os.path.join(out_path, outname, f"train_{i_batch}.pkl")
      save_as_pickle(feature, outpath)
      print(i_batch)
      i_batch+=1
