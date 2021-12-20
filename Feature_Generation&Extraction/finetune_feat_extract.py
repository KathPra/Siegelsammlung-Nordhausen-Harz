### USER INPUT: complete path to image folders
# Attention: Paths must exist already, it cannot be created by this code!
data_dir = "/work-ceph/lprasse/siegel/data/fine_tune" 
outname= "fine_tune_batchsize5_epoch150_Rotated"
out_path = "/work-ceph/lprasse/siegel/models/"


### NO USER INPUT REQUIRED
### Parameters that may be altered: batch_size (change according to memory availability), CUDA/GPU (change according to availability),
### model_name, num_classes, feat_extract, use_pretrained (True/False), num_workers, num_epochs

### CREDIT for code fragments: https://pytorch.org/vision/stable/models.html

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
import torch.optim as optim

### Model settings
device = torch.device("cuda:1")#"cuda:1" or "cpu"
# select from inception, vgg, resnet, alexnet, squeezenet, densenet
model_name = "inception" # resnet #wie tief ist mein Netz?
# Number of classes in the training dataset
num_classes = 4
# Batch size for training
batch_size = 5 #16
# Number of epochs to train for
num_epochs = 150
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    """
    Function defined for model training with train set only.
    """
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
       
        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            if is_inception:
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print(f"Loss: {epoch_loss}")
        print(f"Accuracy: {epoch_acc}")
    return model

def set_parameter_requires_grad(model, feature_extracting):
    """
    Selects the model parameter to update (all / only classification head).
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize the model specified in variable model_name, adjust the classification head to the output classes specified
    in num_classes, and selects the parameter to be updated specified using feat_extract.
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    return model_ft, input_size

### Function calls
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#print(model_ft) # Print the model we just instantiated

# Data augmentation and normalization for training
# zoom in, zoom out
torch.manual_seed(17)
data_transforms = transforms.Compose([
    transforms.Resize((input_size,input_size)), 
    transforms.RandomRotation((0,360)),                                   
    transforms.ToTensor()
])

## Load dataset and create data loader
image_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=25)
print("data loaded")

# Send the model to GPU/CUDA
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Save finetuned/trained model
torch.save(model_ft, os.path.join(out_path, outname))

