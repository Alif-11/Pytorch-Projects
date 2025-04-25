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
                                             
# dataloader for the training and testing datasets - prevents us from having
# to push the entire dataset into RAM.
# we set shuffle to True, so that we don't get caught up in learning loops.
# since we are using stochastic mini batch learning, there is a chance for
# our model to get stuck in some loop of parameter learning. Thus, turning on
# the shuffle mechanism is important.
training_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
testing_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)


class ConvolutionalNeuralNetwork(nn.Module):

  # define the layers of this neural network
  def __init__(self, num_classes):
    super(ConvolutionalNeuralNetwork, self).__init__()
    # 3 in channels, 1 for each of R, G, and B. 
    # 32 out channels, for the 32 hidden layer dimensions

    # input shape (32, 32, 3)
    self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
    # now shape is (30, 30, 32)
    self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
    # now shape is (28, 28, 32)
    self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    # now shape is (14, 14, 32)
    self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    # now shape is (12, 12, 64)
    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
    # now shape is (10, 10, 64)
    self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    # now shape is (5, 5, 64)
    self.conv_layer5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
    # now shape is (3, 3, 64)
    # Flattened value is (572,)

    self.linear1 = nn.Linear(in_features=572,out_features=256)
    # now shape is (256,)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(in_features=256,out_features=num_classes)
    # now shape is (num_classes)

  # define how data is passed through the layers of this neural network
  def forward(self, image):
    conv1_output = self.conv_layer1(image)
    conv2_output = self.conv_layer2(conv1_output)
    max_pool1_output = self.max_pool1(conv2_output)
    conv3_output = self.conv_layer3(max_pool1_output)
    conv4_output = self.conv_layer4(conv3_output)
    max_pool2_output = self.max_pool2(conv4_output)
    conv5_output = self.conv_layer5(max_pool2_output)

    # first dimension is the batch size dimension
    # second dimension is flattened version of the image
    flattened_output = conv5_output.reshape(conv5_output.size(0), -1) 
    linear1_output = self.linear1(flattened_output)
    relu1_output = self.relu1(linear1_output)
    linear2_output = self.linear2(relu1_output)
    return linear2_output
