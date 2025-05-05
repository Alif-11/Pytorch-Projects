import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import deep_cnn
import tqdm
import sys


## training loop hyperparameters
batch_size = 64
num_classes = 10
learning_rate = 0.01
num_epochs = 20

chosen_epoch_idx = 17
chosen_pretrained_weight_suffix = f"epoch{chosen_epoch_idx}"
pretrained_weights_full_path = f"/Users/alifabdullah/Collaboration/Pytorch-Projects/initial-deep-cnn/saved_models/cnn_model_trained_on_cifar10_{chosen_pretrained_weight_suffix}.pth"

use_pretrained = True

do_visualize = False

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
# note that the CIFAR10 dataset only has 10 classes (as per its name)
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


cnn_model = deep_cnn.ConvolutionalNeuralNetwork(num_classes)

# set the loss function to cross entropy using the criterion variable
criterion = nn.CrossEntropyLoss()

# set the optimizer to SGD using the optimizer variable
optimizer = torch.optim.SGD(cnn_model.parameters(),lr=learning_rate,weight_decay=weight_decay, momentum=momentum)

def load_saved_model(full_path_to_saved_model):
  """ Returns an instance of your Deep CNN model, initialized with pretrained weights

  Args:
    full_path_to_saved_model: contains the full path to a set of pretrained_weights (strings)
  """
  cnn_model = deep_cnn.ConvolutionalNeuralNetwork()
  cnn_model.load_state_dict(torch.load(full_path_to_saved_model))

  return cnn_model

if use_pretrained: # if we want to use pretrained weights, then load the pretrained weights
                   # into an appropriate model architecture
  print("Got the pretrained model:", pretrained_weights_full_path)
  cnn_model = load_saved_model(pretrained_weights_full_path)



# Here is the training loop
for epoch_idx in tqdm.tqdm(range(chosen_epoch_idx, num_epochs), desc="Epoch Loop"):
  for idx, (images, labels) in enumerate(tqdm.tqdm(training_loader, desc="Batch Loop")):

    # set the images and labels to the same device (in our case cpu)
    images = images.to(device) # of shape [batch_size, num_channels, height, width]
    labels = labels.to(device) # of shape [batch_size]

    ## shapes look alright
    #


    #cnn_model = deep_cnn.ConvolutionalNeuralNetwork(num_classes)

    # only uncomment these two lines if you want to visualize the cnn layers' outputs
    if do_visualize:
      cnn_model.forward_visualized(images)
      sys.exit(0) # we just want this line and the line above for visualization
    

    cnn_model_outputs = cnn_model(images)
    #print("cnn_model_outputs", cnn_model_outputs.shape)
    cross_entropy_loss = criterion(cnn_model_outputs, labels)

    # optimize the model now
    optimizer.zero_grad() # to prevent gradient accumulation, we zero gradients
    cross_entropy_loss.backward() # backpropagation!
    optimizer.step() # one optimization step

    #print("images shape", images.shape)
    #print("labels shape", labels.shape)
    #print("idx", idx)
    #print("length of training loader", len(training_loader))
  print(f"Finished epoch {epoch_idx} with a loss of {cross_entropy_loss.item()}.")
  torch.save(cnn_model.state_dict(),f"/Users/alifabdullah/Collaboration/Pytorch-Projects/initial-deep-cnn/saved_models/cnn_model_trained_on_cifar10_epoch{epoch_idx+1}.pth")
  