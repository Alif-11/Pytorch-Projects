import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resize images down to 64 x 64
# tensorify them
# normalize,
# random rotate them, then random affine them
image_transforms_composition = transforms.Compose([transforms.Resize((64,64)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                               mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010]),
                                             transforms.RandomRotation(15),
                                             transforms.RandomAffine(10,shear=(-10,10,-10,10))
                                             ])

# get train and test data
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=image_transforms_composition,download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=image_transforms_composition,download=True)
                                             