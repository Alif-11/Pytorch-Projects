Already obtaining some fascinating results from the 3rd epoch trained CNN
on the CIFAR10 dataset.

Original, randomly sampled CIFAR10 image, after applying various transforms:
![Some sort of cat?](model_images/repeating_rainbow_cat.png)
Some kind of cat, rotated??

All channels of the CIFAR10 image, after the first convolutional block:
![Green cats](model_images/green_cat_cloned.png)
A bunch of green cats

I initially thought that my Deep CNN was a little two deep, that I added too many max pooling layers, and it caused it to not perform well on the data.

However, my model output has shape [batch_size, num_classes], while my labels only has shape [batch_size]. However, I put the model output and the labels directly into my criterion. Is this shape mismatch causing the issue? Is the criterion broadcasting labels over my model output, causing such major issues with my loss? I still think I need to reduce how deep my CNN - but what if the bigger issue is the shapes?