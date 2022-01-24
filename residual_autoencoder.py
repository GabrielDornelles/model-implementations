import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import pylab
import matplotlib.pyplot as plt
import albumentations
from PIL import Image
import torch.nn.functional as F
from piqa import SSIM


class GoldenSamplesDataSet(Dataset):
    def __init__(self, image_paths, resize=None):
        self.image_paths = image_paths
        self.resize = resize
        
        # ImageNet mean and std
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
  
        if self.resize is not None:
            image = image.resize(
                self.resize, resample=Image.BILINEAR
            )

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
        }


class AutoEncoder(nn.Module):
    """
    Skip-Connected Convolutional Autoencoder to learn PCB Designs by golden samples.
    reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8347834/
    Denoising variation: https://arxiv.org/pdf/2008.12589.pdf
    """
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

        self.conv1 = nn.Conv2d(in_channels=3 ,  out_channels=64,  kernel_size=(5,5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64 , out_channels=64,  kernel_size=(5,5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=(5,5), stride=1, padding=2)
        self.conv8 = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=(5,5), stride=1, padding=2)
        self.conv9 = nn.Conv2d(in_channels=64,  out_channels=3,   kernel_size=(3,3), stride=1, padding=1)

        self.bn64  = nn.BatchNorm2d(num_features=64)
        self.bn128 = nn.BatchNorm2d(num_features=128)

        self.convtransposed64 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(2,2), stride=2)
        self.convtransposed128 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=(2,2), stride=2)

    def encoder(self, x):
        self._keep_conv1 = self.conv1(x)
        x = F.relu(self.maxpool(self.bn64(self._keep_conv1)))
        self._keep_conv2 = self.conv2(x)
        x = F.relu(self.maxpool(self.bn64(self._keep_conv2)))
        self._keep_conv3 = self.conv3(x)
        x = F.relu(self.maxpool(self.bn128(self._keep_conv3)))
        x = F.relu(self.maxpool(self.bn128(self.conv4(x))))
        x = F.relu(self.bn128(self.conv5(x)))
        return x
    
    def decoder(self, x):
        upsampling1 = self.convtransposed128(x)
        conv6 = F.relu(self.bn128(self.conv6(upsampling1)))
        upsampling2 = self.convtransposed128(conv6) 
        skip1 = upsampling2 + self._keep_conv3
        conv7 = F.relu(self.bn64(self.conv7(skip1)))
        upsampling3 = self.convtransposed64(conv7) 
        skip2 = upsampling3 + self._keep_conv2
        conv8 = F.relu(self.bn64(self.conv8(skip2)))
        upsampling4 = self.convtransposed64(conv8)
        skip3 = upsampling4 + self._keep_conv1
        conv9 = F.relu(self.conv9(skip3))
        return conv9
        
    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

def SSIM_Normalized(x,y):
    """
    @author: gdm, december 2021.
    Normalizes 4d tensor to 0-1 range on its last 2 dimensions.
    Return: Inversed SSIM (1-criterion) to minimize the loss function.
    """
    x =  (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y =  (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    return 1 - criterion(x,y)

model = AutoEncoder()
x_test = torch.rand((1, 3, 400, 400))
output = model(x_test)
print(output.shape)


mse_loss = nn.MSELoss()
device = torch.device("cpu")
criterion = SSIM() # .to(device) if you need GPU support
model.to(device)



# Training

num_epochs = 100
learning_rate = 3.0e-4
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=5e-5)

DATA_DIR = "/home/gabriel/Documents/golden_samples"
image_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

train_dataset = GoldenSamplesDataSet(image_paths=image_files, resize=(400,400)) #370 250
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)

losses = np.zeros(num_epochs)

for epoch in range(num_epochs):
    i = 0
    for inputs in train_loader:
        x = inputs["images"]
        x = x.to(device)
        xhat = model(x)
        #loss = SSIM_Normalized(x,xhat)
        loss = mse_loss(xhat, x)
        loss.backward()
        losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
        optimizer.zero_grad()
        optimizer.step()
        i += 1

    plt.figure()
    pylab.xlim(0, num_epochs)
    plt.plot(range(0, num_epochs), losses, label='loss')
    plt.legend()
    #plt.savefig(os.path.join("/save/", 'loss.pdf'))
    plt.close()

    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
         loss))
         
# visualize

#def pre_process(frame):
#print(train_dataset.__getitem__(0)["images"].shape)
image = Image.open("/home/gabriel/Documents/golden_samples/IMG_20211211_112703.jpg").convert('RGB')
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True)
            ]
        )
image = image.resize(
    (400,400), resample=Image.BILINEAR
)

image = np.array(image)
augmented = aug(image=image)
image = augmented["image"]
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float)
image = image[None,:,:,:]
image = image.to(device)
#image.shape
output = model(image)
output = output.squeeze(0)
output = output.transpose(2,0)
output.shape
test_model = output.detach().numpy()

test_model = Image.fromarray(test_model, 'RGB')
#display(test_model)
test_model.save("/home/gabriel/Documents/aa.png")
