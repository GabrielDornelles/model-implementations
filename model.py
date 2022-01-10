
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from torchmetrics.functional import accuracy

class LitResNet(pl.LightningModule):

	def __init__(self):
		super().__init__()
		self.model = models.resnet18(pretrained=False, num_classes=100)
		self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)	
		self.model.maxpool = nn.Identity()

	def forward(self, x):
		out = self.model(x)
		return F.log_softmax(out, dim=1)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer
	
	def evaluate(self, batch, stage = None):
		x, y = batch
		logits = self(x)
		loss = F.nll_loss(logits, y)
		preds = torch.argmax(logits, dim=1)
		acc = accuracy(preds, y)

		if stage:
			self.log(f"{stage}_loss", loss, prog_bar=True)
			self.log(f"{stage}_acc", acc, prog_bar=True)

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		out = self(x)
		loss = F.nll_loss(out,y)
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		self.evaluate(batch, "val")

	def test_step(self, batch, batch_idx):
		# there's no test but if you want simple call evaluate like that
		self.evaluate(batch, "test")

# data
dataset = CIFAR100(root='./',train=True, download=True, transform=transforms.ToTensor())
train_size = int(len(dataset) * 0.9)
val_size = int(len(dataset) - train_size)
cifar_train, cifar_val = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(cifar_train, batch_size=32, num_workers=4)
val_loader = DataLoader(cifar_val, batch_size=32, num_workers=4)

# model
model = LitResNet()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.01, callbacks=RichProgressBar())
trainer.fit(model, train_loader, val_loader)
    