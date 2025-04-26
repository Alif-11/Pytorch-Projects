import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from . import deep_cnn

## training loop hyperparameters
batch_size = 64
num_classes = 10
learning_rate = 0.01
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## optimizer hyperparameters
weight_decay = 0.01
momentum = 0.9

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
                                             
# dataloader for the training and testing datasets - prevents us from having
# to push the entire dataset into RAM.
# we set shuffle to True, so that we don't get caught up in learning loops.
# since we are using stochastic mini batch learning, there is a chance for
# our model to get stuck in some loop of parameter learning. Thus, turning on
# the shuffle mechanism is important.
training_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
testing_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


cnn_model = deep_cnn.ConvolutionalNeuralNetwork()

# set the loss function to cross entropy using the criterion variable
criterion = nn.CrossEntropyLoss()

# set the optimizer to SGD using the optimizer variable
optimizer = torch.optim.SGD(cnn_model.parameters(),lr=learning_rate,weight_decay=weight_decay, momentum=momentum)
