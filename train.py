import torch.optim as optim
from model import Net
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from model import model
from rich.progress import track
import numpy as np
import copy
import os
import time
from rich import print
# check https://rich.readthedocs.io/en/latest/appendix/colors.html?highlight=colors to use colors you want on displaying data
# hyperparameters
epoches=25
batch_size=128


## -> dataset
## ImageNet normalization, static
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.485, 0.406), (0.229, 0.224, 0.225))])

# feels like its working
image_datasets = {x: torchvision.datasets.CIFAR10(root='./data', train=True,download=True if x=="train" else False, transform=transform) for x in ['train','val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"classes: {class_names}")

## -> net and training
criterion = nn.CrossEntropyLoss()
model = model()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)# Stochastic Gradient Descent (SGD)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
start = time.time()
for epoch in range(epoches):  # loop over the dataset multiple times
    print(f"\nepoch: {epoch} / {(epoches-1)}")
    print("----------")
    for phase in ["train","val"]:
        if phase == "train":
            model.train()
            description="[bold gold3]Training...  [/bold gold3]"
        else:
            model.eval()
            description="[bold blue_violet]Validating...[/bold blue_violet]"
        running_loss = 0.0
        running_corrects = 0
        for inputs,labels in track(dataloaders[phase],description=description):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            ## --> forward step
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds= torch.max(outputs,1)
                loss = criterion(outputs,labels)# calculates loss its picks the right labels and calculate based on the output predicted
            # --> backward step
            if phase== "train":
                loss.backward()
                optimizer.step()

            ## --> statistics
            running_loss += running_loss * inputs.size(0) # batchsize
            running_corrects += torch.sum(preds==labels.data)# I think preds is a batch of 4 inputs, and we compare the sum of statements that matches the labels.data
            
            # float in python are C doubles, .double is REAL double i guess
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =running_corrects.double()/ dataset_sizes[phase]
            
        print(f"[bold]Loss[/bold] : {loss:.4f}   [bold]Acc[/bold]: {epoch_acc:.4f}")
        if phase=="val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
end = time.time()
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(),"cifar-10_model.pth")
print(f"time elapsed: {(end-start):.2f}")
print('Finished Training')
