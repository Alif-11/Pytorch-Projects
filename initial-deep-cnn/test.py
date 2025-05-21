import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import deep_cnn
import tqdm
import sys
import os

batch_size = 64
num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transforms_composition = transforms.Compose([transforms.Resize((64,64)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                               mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010]),
                                             transforms.RandomRotation(15),
                                             transforms.RandomAffine(10,shear=(-10,10,-10,10))
                                             ])

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=image_transforms_composition,download=True)
testing_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

chosen_pretrained_weights_full_path = "/Users/alifabdullah/Collaboration/Pytorch-Projects/initial-deep-cnn/saved_models/cnn_model_trained_on_cifar10_epoch20.pth"

test_cnn_model = deep_cnn.ConvolutionalNeuralNetwork(num_classes=num_classes)
test_cnn_model.load_state_dict(torch.load(chosen_pretrained_weights_full_path, map_location=device))
test_cnn_model.to(device)
test_cnn_model.eval()

count = 0
correct = 0

with torch.no_grad():
  for idx, (images, labels) in enumerate(tqdm.tqdm(testing_loader, desc="Evaluation Loop")):
    images = images.to(device) 
    labels = labels.to(device) 

    cnn_model_outputs = test_cnn_model(images)
    print(cnn_model_outputs.shape)
    print(labels.shape)
    print("End of loop")
    break