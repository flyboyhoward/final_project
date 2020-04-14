#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from urllib.request import urlopen
# Plot ad hoc data instances
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image as PImage

import torchbearer
from torchbearer import Trial

validation_split = .2
shuffle_dataset = True
random_seed= 42
# the number of images that will be processed in a single step
batch_size=128
# the size of the images that we'll learn on - we'll shrink them from the original size for speed
image_size=(224, 224)

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
])
dataset = ImageFolder("data/train", transform)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print('device: ', device)

def train():
	# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	print('start training')
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
	                                           sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                                sampler=valid_sampler)

	model = resnet50(pretrained=True)
	model.avgpool = nn.AdaptiveAvgPool2d((1,1))
	model.fc = nn.Linear(2048, len(dataset.classes))
	model.train()

	# Freeze layers by not tracking gradients
	for param in model.parameters():
	    param.requires_grad = False
	model.fc.weight.requires_grad = True #unfreeze last layer weights
	model.fc.bias.requires_grad = True #unfreeze last layer biases

	optimiser1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3) #only optimse non-frozen layers
	loss_function = nn.CrossEntropyLoss()
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	print(device)
	checkpointer = torchbearer.callbacks.checkpointers.Best(filepath='model.pt', monitor='loss')

	trial = Trial(model, optimiser1, loss_function, metrics=['loss', 'accuracy'], callbacks=[checkpointer]).to(device)
	trial.with_generators(train_loader, val_generator=val_loader)
	trial.run(epochs=10)

	state_dict = torch.load('model.pt')
	trial_reloaded = Trial(model, optimiser1, loss_function, metrics=['loss', 'accuracy'], callbacks=[checkpointer]).to(device)
	trial_reloaded.with_generators(train_loader, val_generator=val_loader)
	trial_reloaded.load_state_dict(state_dict)
	trial_reloaded.run(epochs=20)

	results = trial_reloaded.evaluate(data_key=torchbearer.VALIDATION_DATA)
	print()
	print(results)
	torch.save(model, 'mymodel.pth')


def test(img):
	model = torch.load('mymodel.pth')
	model.eval()
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	classes = dataset.classes

	pil_img = PImage.fromarray(img)
	val_pred = model(transform(pil_img).unsqueeze(0).to(device)).data.cpu().numpy()
	# print(val_pred)
	val_pred_max = val_pred.max()
	if abs(val_pred_max) < 1:
		pred = 'onroad'
	else:
		pred = classes[val_pred.argmax()]
	
	return pred

def main():
	print('running as  main function, now begin training')
	train()

if __name__ == '__main__':
	main()


