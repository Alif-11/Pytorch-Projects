import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

class ConvolutionalNeuralNetwork(nn.Module):

  
  # initially used this class for the CIFAR10 dataset, which is why we have a
  # default value of 10 for our num_classes
  def __init__(self, num_classes=10):

    """ Define the layers of this Convolutional Neural Network. 

    Args:
      num_classes: number of output classes (int)
    """

    super(ConvolutionalNeuralNetwork, self).__init__()
    # 3 in channels, 1 for each of R, G, and B. 
    # 32 out channels, for the 32 hidden layer dimensions

    # input shape [per image] is (4, 64, 64)
    self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
    # now shape is (32, 62, 62)
    self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
    # now shape is (32, 60, 60)
    self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    # now shape is (32, 30, 30)
    self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    # now shape is (32, 28, 28)
    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
    # now shape is (64, 26, 26)
    self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    # now shape is (64, 13, 13)
    self.conv_layer5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
    # now shape is (64, 11, 11)
    self.conv_layer6 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3)
    # now shape is (32, 9, 9)
    self.max_pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
    # now shape is (32, 4, 4)
    
    # Flattened value is now (512,)

    self.linear1 = nn.Linear(in_features=512,out_features=256)
    # now shape is (256,)
    self.linear2 = nn.Linear(in_features=256,out_features=128)
    # now shape is (128,)
    self.relu1 = nn.ReLU()
    self.linear3 = nn.Linear(in_features=128,out_features=num_classes)
    # now shape is (num_classes)

  # define how data is passed through the layers of this neural network
  def forward(self, image):
    #print("im shape", image.shape)
    conv1_output = self.conv_layer1(image)
    #print("conv1_output.shape", conv1_output.shape)
    conv2_output = self.conv_layer2(conv1_output)
    #print("conv2_output.shape", conv2_output.shape)
    max_pool1_output = self.max_pool1(conv2_output)
    #print("max_pool1_output.shape", max_pool1_output.shape)
    conv3_output = self.conv_layer3(max_pool1_output)
    #print("conv3_output.shape", conv3_output.shape)
    conv4_output = self.conv_layer4(conv3_output)
    #print("conv4_output.shape", conv4_output.shape)
    max_pool2_output = self.max_pool2(conv4_output)
    #print("max_pool2_output.shape", max_pool2_output.shape)
    conv5_output = self.conv_layer5(max_pool2_output)
    #print("conv5_output.shape", conv5_output.shape)
    conv6_output = self.conv_layer6(conv5_output)
    #print("conv6_output.shape", conv6_output.shape)
    max_pool3_output = self.max_pool3(conv6_output)
    #print("max_pool3_output.shape", max_pool3_output.shape)

    # first dimension is the batch size dimension
    # second dimension is flattened version of the image
    flattened_output = max_pool3_output.reshape(max_pool3_output.shape[0], -1) 
    linear1_output = self.linear1(flattened_output)
    linear2_output = self.linear2(linear1_output)
    relu1_output = self.relu1(linear2_output)
    linear3_output = self.linear3(relu1_output)
    return linear3_output

  def forward_visualized(self, image):
    """ A version of the forward call that visualizes the output of each step of the Deep CNN Model I made.
    """

    #print("im shape", image.shape)
    image_reshaped = torch.reshape(image[0], (64,64,3))

    # Scale the pixel values to the range 0-255 and convert to uint8
    image_scaled = np.array((image_reshaped * 255).to(torch.uint8))
    cv2.imshow("First CIFAR10 Image",image_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    conv1_output = self.conv_layer1(image)
    # Transpose the image to the correct format (height, width, channels)
    
    #print("conv1_output.shape", conv1_output.shape)
    conv2_output = self.conv_layer2(conv1_output)
    #print("conv2_output.shape", conv2_output.shape)
    max_pool1_output = self.max_pool1(conv2_output)
    #print("max_pool1_output.shape", max_pool1_output.shape)
    conv3_output = self.conv_layer3(max_pool1_output)
    #print("conv3_output.shape", conv3_output.shape)
    conv4_output = self.conv_layer4(conv3_output)
    #print("conv4_output.shape", conv4_output.shape)
    max_pool2_output = self.max_pool2(conv4_output)
    #print("max_pool2_output.shape", max_pool2_output.shape)
    conv5_output = self.conv_layer5(max_pool2_output)
    #print("conv5_output.shape", conv5_output.shape)
    conv6_output = self.conv_layer6(conv5_output)
    #print("conv6_output.shape", conv6_output.shape)
    max_pool3_output = self.max_pool3(conv6_output)
    #print("max_pool3_output.shape", max_pool3_output.shape)

    # first dimension is the batch size dimension
    # second dimension is flattened version of the image
    flattened_output = max_pool3_output.reshape(max_pool3_output.shape[0], -1) 
    linear1_output = self.linear1(flattened_output)
    linear2_output = self.linear2(linear1_output)
    relu1_output = self.relu1(linear2_output)
    linear3_output = self.linear3(relu1_output)
    return linear3_output
