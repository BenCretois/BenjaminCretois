---
title: Building a CNN from scratch with PyTorch
author: ''
date: '2021-05-05'
slug: building-a-cnn-from-scratch-with-pytorch
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-05-05T12:43:49+02:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

In this tutorial I will introduce how to build a **Convolutional Neural Network (CNN)** from scratch with a real dataset (most tutorials use easy to use dataset such as MNIST or CIFAR-10). As we will see, it is fine to build your own model but it is sometimes more efficient to use pre-existing ones. Thus, I will also introduce you to fine-tuning a pre-trained CNN.

In this tutorial we will use a dataset composed of pandas, dogs and cats! Before getting into the code make sure you download the dataset at this link: https://zenodo.org/record/4738462#.YJJIH7UzYuU. 

First of all let's load the libraries we will use in this tutorial:


```python
# Libraries and modules that will help set and train the model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# Libraries that will help building the dataset
import torchvision
from torchvision import datasets, models, transforms

# For visualisation
import matplotlib.pyplot as plt
%matplotlib inline

# Helper libraries
import time
import copy

# To manipulate path
import os
import shutil
from random import shuffle
```


As you have noticed, the data is composed of a folder *images* containing **three subfolders**: cats, dogs and pandas. The very first step is to build a training and testing dataset (and a validation set that we will skip here for the sake of simplicity). Here is a function that divide the images folder into a training / testing split and that copy the .png files into the training / testing subfolders.

Note that here we use a 90 / 10 split, meaning that 90% of the images will be used for training the model while 10% will be used to test it.


```python
def train_test_split(im_dir, train_dir, test_dir):
    
    # List the images from all folders
    list_dir = os.listdir(im_dir)

    # Shuffle in case the folder is not randomly distributed
    shuffle(list_dir)

    # Compute the length of the directory
    len_dir = len(list_dir)

    # Set the number of training images to sample
    training_num = int(0.9*len_dir)

    # Sample a training and testing dataset:

    # Take images from 0 to training_num
    training = list_dir[:training_num]

    # Take images from training_num to the end
    testing = list_dir[training_num:]

    # Save the images in a dedicated folder
    for image_name in training:
        image_path = os.path.join(im_dir, image_name)
        image_out_path = os.path.join(train_dir, image_name)
        shutil.copyfile(image_path, image_out_path)

    for image_name in testing:
        image_path = os.path.join(im_dir, image_name)
        image_out_path = os.path.join(test_dir, image_name)
        shutil.copyfile(image_path, image_out_path)
```

We set the path to the images, training and test directories for all three species. Note that I usually place all these path in a **config.py** file, this makes the script less messy.


```python
# Set the path to the images dir
cat_dir = "data/images/cats"
dog_dir = "data/images/dogs"
panda_dir = "data/images/pandas"

# Set the path to the training dir
cat_train = "data/train/cats"
dog_train = "data/train/dogs"
panda_train = "data/train/pandas"

# Set the path to the testing dir
cat_test = "data/test/cats"
dog_test = "data/test/dogs"
panda_test = "data/test/pandas"

# Set the path for the train and test folder
train_path = "data/train"
test_path = "data/test"
```

And then we populate our subfolders using the train_test_split function:


```python
train_test_split(cat_dir, cat_train, cat_test)
train_test_split(dog_dir, dog_train, dog_test)
train_test_split(panda_dir, panda_train, panda_test)
```

We will normalize the images by calculating their mean and standard deviation. It has been shown that normalisation can drastically improve the performance of neural networks. 

I did not find any straightforward way to calculate these descriptive statistics so I defined a function that first resize the images and turn them into tensors. The function then lists all the images in the folder and initiate a data loader that will returns batches of 100 images. For each batch the function calculates the mean and the std. Finally the function returns the mean of the batches means and standard deviation.

**WARNING**: Be careful though, the mean of different standard deviation is an approximation of the dataset's standard deviation!

**Important to note**: The testing data will be normalized based on the mean and std of the training dataset, otherwise there will be a leakage of information between the training and testing dataset!

**Other note**: usually I will store the mean and standard deviation in a JSON file so I don't need to recompute them every time I run a script, as you can see I would need to go through the whole training dataset at every run.

First we define our function that will calculate the mean and std:


```python
def mean_std(path):

    # Some work on the images first
    trans = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    # List the pictures in the training folder
    train_dataset = torchvision.datasets.ImageFolder(root=path,
                                              transform=trans)
    loader_mean = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=100,
                                                   shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader_mean:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # collapse the 2 last dim into a single one
        mean += data.mean(2).sum(0) # the mean is computed across the 3 channels
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return(mean, std)
```

Finally we calculate the mean and standard deviation of the dataset:


```python
mean, std = mean_std(train_path)
print(mean, std)
```

    tensor([0.4585, 0.4360, 0.3906]) tensor([0.2267, 0.2222, 0.2190])


Now that we got the mean and standard deviation of the training dataset we can initiate the transormation that will be used on the dataset (we resize, turn into a tensor and normalize our training and testing dataset)


```python
trans = transforms.Compose([
    # Out neural network will take images that are 64 x 64 pixels
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

Here we list all the images in the training and testing folders.


```python
train_dataset = torchvision.datasets.ImageFolder(root="data/train/", transform=trans)
test_dataset = torchvision.datasets.ImageFolder(root="data/test/", transform=trans)
```

The `images` folders should be have a strict structure as the name of the subfolders will be used as a label. For instance, in the subfolder *test* I have 3 subfolders: *cats*, *dogs*, *pandas*. The name of these 3 subfolders will be used as the label. `torchvision.datasets.ImageFolder` also turns these labels into numbers (here 0 for cats, 1 for dogs and 2 for pandas). The `torchvision.datasets.ImageFolder` comes handy as it returns the transformed image and its label as a tuple:


```python
img, label = train_dataset[1]
```


```python
test_dataset.class_to_idx
```




    {'cats': 0, 'dogs': 1, 'pandas': 2}



Next, we initiate the dataloaders. The dataloaders basically return a batch of images (here batches of 32) so the model don't have to do a Gradient Descent on the full dataset (this is what we call **Stochastic Gradient Descent** see this [video](https://www.youtube.com/watch?v=vMh0zPT0tLI) for a solid introduction).


```python
dataloader_train = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True, num_workers=4)

dataloader_test = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=32,
                                               shuffle=True, num_workers=4)
```

It is now time to check if our pipeline is working as it should by quickly visualizing some images. We will define a function that will return a batch of images along with their label.


```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
```

Next we basically return a batch of images. **dataiter** is an object that iterates over the dataloader, the method `next()` then does the iteration.


```python
dataiter = iter(dataloader_train)
images, labels = dataiter.next()
```

Note the shape of the `images` tensor: we have a tensor of size [32, 3, 64, 64]. This means that in our case the tensor contains 32 images (batch size) that have 3 channels (Red Blue Green) and each image is 64x64 pixels. This is important to keep this shape as a PyTorch model is very picky with what it's being fed with!


```python
images.shape
```




    torch.Size([32, 3, 64, 64])



Finally we plot the images using the handy `make_grid` method from `torchvision.make_grid`:


```python
# Make a grid from batch
out = torchvision.utils.make_grid(images)
imshow(out)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![](/post/2021-05-05-building-a-cnn-from-scratch-with-pytorch/tutorial_28_1.png)
    


# Training a CNN from scratch

Here we are! We finally made it to the step of building our neural network. Surprisingly, this step is usually fast compared to the data preparation phase. In this first part we will train a CNN in PyTorch from scratch, meaning that we will specify all layers and will go through the mechanics of PyTorch.

First of all, I create an object `SummaryWriter`. This will basically create file and add summaries and events of the model training to it. This is particularly useful when the model takes a long time to train and we want to vizualise its training in near real-time. The written files can then be vizualize in the amazing **Tensorboard**.


```python
writer = SummaryWriter()
```

Next, the heart of the script: the CNN model itself. We will define a simple neural network that has **3 convolutional layers** and **2 fully connected layers**. 

In PyTorch, the model we specify should be a subclass of `nn.Module` (i.e. the model we specify will inherent different properties from the `nn.Module` class). Hence the `super` call. The `super` call delegates the function call to the parent class, which is `nn.Module` in our case.




```python
class ConvNet(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        # Start with 3 channels and output 32 channels
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, kernel_size=3, padding=1)

        # fully connected layers
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        
        # The last fully connected layer should have the number of class
        # has its output
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)

        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return(out)
```


```python
model = ConvNet()
```

We have defined the model but not how the model will learn (or estimates its parameters). Yes, learning is basically parameter estimation, nothing magic in that! For that we need 2 things: a **loss function** and an **optimizer**.

- **The loss function** is used to compare the outputs of our model to the desired output (the targets). Here we use the Cross Entropy loss as we have 3 categories. 
- **The optimizer** is an algorithm or a method used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses. Here we use the Stochastic Gradient Descent algorithm. We will go for a learning rate of 1.e-3 as I have found it works quite well for our specific problem. Note that you may try different learning rate for better performance!


```python
# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=1.e-3)
```

Finally, we define the training loop. In PyTorch, in contrast to Tensorflow, it is very easy to custom the training loop so we can compute custom metrics. Moreover, I personally prefer this style of writing down all steps as it helps me understand the process of training a neural network.

The training loop will loop through each **epochs**. The number of epochs defines the number of times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch can be comprised of one or more batches as this is the case for us (we feed the algorithm with batches of 32 images).


```python
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, writer):

    # Loop through the number of epochs: number of times the algorithm will go through
    # the entire dataset.
    for epoch in range(n_epochs):
        
        # Initiate the metrics for the training set
        loss_train = 0.0
        total_train = 0
        correct_train = 0
        accuracy_train = 0

        # Then loop through a batch of images and labels
        for imgs, labels in train_loader:

            # Feed the model with the tensors
            outputs = model(imgs)
            
            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Freeze the optimizer so it doesn't update yet!
            optimizer.zero_grad()
            
            # Compute the gradients
            loss.backward()
            
            # Update the parameters of the neural network
            optimizer.step()

        ### Accuracy and loss for training and testing ###
            
            # We compute the total loss over the epoch
            loss_train += loss.item()

            # Take the predicted label (category with the highest probability)
            _, predicted = torch.max(outputs, dim=1)
            
            # Store the number of images the number has seen
            total_train += labels.shape[0]
            
            # Finally compute accuracy for the training set
            correct_train += int((predicted == labels).sum())
            accuracy_train = correct_train / total_train

        # Model in evaluation mode -> "turn off" certain layers for
        # inference
        model.eval()
        
        # Initiate the metrics for the testing set
        loss_test = 0.0
        total_test = 0
        correct_test = 0
        accuracy_test = 0

        # turn off gradient computation
        with torch.no_grad():
            
            # Loop through the test images
            for imgs_test, labels_test in test_loader:
                
                # Feed the model with the test images
                outputs_test = model(imgs_test)
                
                # Compute the loss of the test
                loss_test = loss_fn(outputs_test, labels_test)

                # Get the predicted label (the category with the highest probability)
                _, predicted_test = torch.max(outputs_test, dim=1)
                
                # Store the number of test pictures
                total_test += labels_test.shape[0]
                
                # Compute the metric
                correct_test += int((predicted_test == labels_test).sum())
                accuracy_test = correct_test / total_test

        print("Epoch {} / {}, Training loss {}, Train acc {}, Testing loss {}, Test acc {}".format(epoch + 1,
                                                               n_epochs, loss_train, accuracy_train,
                                                                         loss_test, accuracy_test))

        # Add the training metrics to the writer so we can vizualise in TensorBoard
        writer.add_scalar("Loss_train", loss_train, epoch)
        writer.add_scalar("Loss_test", loss_test, epoch)
        writer.add_scalar("Train_acc", accuracy_train, epoch)
        writer.add_scalar("Test_acc", accuracy_test, epoch)

        # Model go back to training mode again!
        model.train()
```

We finally run our training loop!


```python
training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=dataloader_train,
    test_loader=dataloader_test,
    writer=writer
)
```

    Epoch 1 / 100, Training loss 92.97404968738556, Train acc 0.3340740740740741, Testing loss 1.0874452590942383, Test acc 0.3333333333333333
    Epoch 2 / 100, Training loss 92.73141467571259, Train acc 0.3437037037037037, Testing loss 1.0895224809646606, Test acc 0.3466666666666667
    Epoch 3 / 100, Training loss 92.3821827173233, Train acc 0.36703703703703705, Testing loss 1.064963459968567, Test acc 0.37666666666666665
    Epoch 4 / 100, Training loss 91.94769525527954, Train acc 0.40185185185185185, Testing loss 1.065438985824585, Test acc 0.39
    Epoch 5 / 100, Training loss 91.42513906955719, Train acc 0.4274074074074074, Testing loss 1.0634416341781616, Test acc 0.43333333333333335
    Epoch 6 / 100, Training loss 90.75135934352875, Train acc 0.44925925925925925, Testing loss 1.0755324363708496, Test acc 0.47333333333333333
    Epoch 7 / 100, Training loss 89.83235740661621, Train acc 0.4803703703703704, Testing loss 1.0814121961593628, Test acc 0.5033333333333333
    Epoch 8 / 100, Training loss 88.76071053743362, Train acc 0.502962962962963, Testing loss 0.9770997166633606, Test acc 0.5266666666666666
    Epoch 9 / 100, Training loss 87.44006955623627, Train acc 0.5144444444444445, Testing loss 1.0378440618515015, Test acc 0.55
    Epoch 10 / 100, Training loss 85.86386460065842, Train acc 0.5174074074074074, Testing loss 0.988680899143219, Test acc 0.5366666666666666
    Epoch 11 / 100, Training loss 84.17928719520569, Train acc 0.5244444444444445, Testing loss 0.8906807899475098, Test acc 0.5666666666666667
    Epoch 12 / 100, Training loss 82.41291415691376, Train acc 0.5314814814814814, Testing loss 0.9499918818473816, Test acc 0.54
    Epoch 13 / 100, Training loss 81.03854596614838, Train acc 0.5255555555555556, Testing loss 0.878519594669342, Test acc 0.56
    Epoch 14 / 100, Training loss 79.72567760944366, Train acc 0.5325925925925926, Testing loss 1.1437790393829346, Test acc 0.5433333333333333
    Epoch 15 / 100, Training loss 78.73671513795853, Train acc 0.5396296296296297, Testing loss 0.9434453845024109, Test acc 0.56
    Epoch 16 / 100, Training loss 78.08663499355316, Train acc 0.5362962962962963, Testing loss 0.7397613525390625, Test acc 0.5433333333333333
    Epoch 17 / 100, Training loss 77.49842131137848, Train acc 0.5403703703703704, Testing loss 0.8601131439208984, Test acc 0.55
    Epoch 18 / 100, Training loss 77.03634721040726, Train acc 0.5451851851851852, Testing loss 0.7876469492912292, Test acc 0.57
    Epoch 19 / 100, Training loss 76.30121541023254, Train acc 0.552962962962963, Testing loss 0.756324827671051, Test acc 0.5633333333333334
    Epoch 20 / 100, Training loss 76.16341537237167, Train acc 0.5596296296296296, Testing loss 0.9674699902534485, Test acc 0.56
    Epoch 21 / 100, Training loss 75.61177444458008, Train acc 0.5622222222222222, Testing loss 0.9966593384742737, Test acc 0.5633333333333334
    Epoch 22 / 100, Training loss 75.29715085029602, Train acc 0.5622222222222222, Testing loss 0.8943412899971008, Test acc 0.5733333333333334
    Epoch 23 / 100, Training loss 75.27383548021317, Train acc 0.57, Testing loss 0.9753305912017822, Test acc 0.5666666666666667
    Epoch 24 / 100, Training loss 74.90045017004013, Train acc 0.5648148148148148, Testing loss 0.8764877319335938, Test acc 0.5733333333333334
    Epoch 25 / 100, Training loss 74.66580802202225, Train acc 0.5677777777777778, Testing loss 1.11214017868042, Test acc 0.5666666666666667
    Epoch 26 / 100, Training loss 74.56834578514099, Train acc 0.5644444444444444, Testing loss 0.9833900928497314, Test acc 0.5733333333333334
    Epoch 27 / 100, Training loss 74.20974206924438, Train acc 0.5725925925925925, Testing loss 0.7479797005653381, Test acc 0.56
    Epoch 28 / 100, Training loss 74.0707134604454, Train acc 0.5714814814814815, Testing loss 0.9737188816070557, Test acc 0.5633333333333334
    Epoch 29 / 100, Training loss 73.78421151638031, Train acc 0.5725925925925925, Testing loss 0.8975214958190918, Test acc 0.5666666666666667
    Epoch 30 / 100, Training loss 73.75267589092255, Train acc 0.5762962962962963, Testing loss 0.822583019733429, Test acc 0.5933333333333334
    Epoch 31 / 100, Training loss 73.48732686042786, Train acc 0.5748148148148148, Testing loss 0.8835160136222839, Test acc 0.6
    Epoch 32 / 100, Training loss 73.37810844182968, Train acc 0.5785185185185185, Testing loss 0.8305863738059998, Test acc 0.5933333333333334
    Epoch 33 / 100, Training loss 72.90660637617111, Train acc 0.5777777777777777, Testing loss 0.9096582531929016, Test acc 0.5866666666666667
    Epoch 34 / 100, Training loss 72.65916442871094, Train acc 0.59, Testing loss 0.9881632328033447, Test acc 0.5766666666666667
    Epoch 35 / 100, Training loss 72.43601006269455, Train acc 0.5751851851851851, Testing loss 1.0281740427017212, Test acc 0.5933333333333334
    Epoch 36 / 100, Training loss 72.3031656742096, Train acc 0.587037037037037, Testing loss 0.7453406453132629, Test acc 0.5833333333333334
    Epoch 37 / 100, Training loss 71.80919367074966, Train acc 0.5881481481481482, Testing loss 0.7325885891914368, Test acc 0.6
    Epoch 38 / 100, Training loss 71.95956736803055, Train acc 0.5996296296296296, Testing loss 0.6134034991264343, Test acc 0.56
    Epoch 39 / 100, Training loss 71.62563014030457, Train acc 0.5929629629629629, Testing loss 0.9679448008537292, Test acc 0.5933333333333334
    Epoch 40 / 100, Training loss 71.29766798019409, Train acc 0.5988888888888889, Testing loss 0.9043577313423157, Test acc 0.5766666666666667
    Epoch 41 / 100, Training loss 71.17770326137543, Train acc 0.6003703703703703, Testing loss 0.9632486701011658, Test acc 0.6033333333333334
    Epoch 42 / 100, Training loss 70.70167261362076, Train acc 0.6029629629629629, Testing loss 0.6671697497367859, Test acc 0.5966666666666667
    Epoch 43 / 100, Training loss 70.4370933175087, Train acc 0.6040740740740741, Testing loss 0.5890529751777649, Test acc 0.5966666666666667
    Epoch 44 / 100, Training loss 70.0095773935318, Train acc 0.6066666666666667, Testing loss 0.8609921932220459, Test acc 0.5933333333333334
    Epoch 45 / 100, Training loss 70.17636954784393, Train acc 0.607037037037037, Testing loss 0.7703410983085632, Test acc 0.61
    Epoch 46 / 100, Training loss 69.72288757562637, Train acc 0.61, Testing loss 0.8842969536781311, Test acc 0.6033333333333334
    Epoch 47 / 100, Training loss 69.47107654809952, Train acc 0.6111111111111112, Testing loss 0.8496959805488586, Test acc 0.6033333333333334
    Epoch 48 / 100, Training loss 68.91402232646942, Train acc 0.6159259259259259, Testing loss 0.8802845478057861, Test acc 0.61
    Epoch 49 / 100, Training loss 68.69548678398132, Train acc 0.6144444444444445, Testing loss 0.8952004909515381, Test acc 0.5966666666666667
    Epoch 50 / 100, Training loss 68.75062638521194, Train acc 0.6225925925925926, Testing loss 0.8620046973228455, Test acc 0.6033333333333334
    Epoch 51 / 100, Training loss 68.22007900476456, Train acc 0.6218518518518519, Testing loss 0.9135915637016296, Test acc 0.6166666666666667
    Epoch 52 / 100, Training loss 68.1612748503685, Train acc 0.6259259259259259, Testing loss 0.9867925047874451, Test acc 0.6066666666666667
    Epoch 53 / 100, Training loss 67.82372635602951, Train acc 0.6281481481481481, Testing loss 0.9717295169830322, Test acc 0.6066666666666667
    Epoch 54 / 100, Training loss 67.38524234294891, Train acc 0.6285185185185185, Testing loss 0.6347225308418274, Test acc 0.5933333333333334
    Epoch 55 / 100, Training loss 67.28690314292908, Train acc 0.6288888888888889, Testing loss 0.9006388783454895, Test acc 0.6133333333333333
    Epoch 56 / 100, Training loss 66.71102058887482, Train acc 0.6322222222222222, Testing loss 0.8623394966125488, Test acc 0.6133333333333333
    Epoch 57 / 100, Training loss 66.55319648981094, Train acc 0.6337037037037037, Testing loss 0.835072934627533, Test acc 0.6133333333333333
    Epoch 58 / 100, Training loss 66.18024504184723, Train acc 0.6377777777777778, Testing loss 0.617293655872345, Test acc 0.6066666666666667
    Epoch 59 / 100, Training loss 65.77396476268768, Train acc 0.6314814814814815, Testing loss 0.8882052302360535, Test acc 0.6033333333333334
    Epoch 60 / 100, Training loss 65.69327169656754, Train acc 0.6344444444444445, Testing loss 0.6915426254272461, Test acc 0.6133333333333333
    Epoch 61 / 100, Training loss 65.2914245724678, Train acc 0.6374074074074074, Testing loss 0.8516831994056702, Test acc 0.6233333333333333
    Epoch 62 / 100, Training loss 64.96074956655502, Train acc 0.64, Testing loss 0.5304060578346252, Test acc 0.6066666666666667
    Epoch 63 / 100, Training loss 64.33942584693432, Train acc 0.6459259259259259, Testing loss 0.763451099395752, Test acc 0.6266666666666667
    Epoch 64 / 100, Training loss 63.98579388856888, Train acc 0.6514814814814814, Testing loss 1.0976688861846924, Test acc 0.6266666666666667
    Epoch 65 / 100, Training loss 63.61524644494057, Train acc 0.6533333333333333, Testing loss 0.9439403414726257, Test acc 0.6166666666666667
    Epoch 66 / 100, Training loss 63.563642263412476, Train acc 0.6503703703703704, Testing loss 0.5973674654960632, Test acc 0.63
    Epoch 67 / 100, Training loss 63.18000739812851, Train acc 0.6548148148148148, Testing loss 0.5893906950950623, Test acc 0.64
    Epoch 68 / 100, Training loss 62.86909991502762, Train acc 0.6514814814814814, Testing loss 0.5968372225761414, Test acc 0.6233333333333333
    Epoch 69 / 100, Training loss 62.546015202999115, Train acc 0.6555555555555556, Testing loss 0.7776437401771545, Test acc 0.6566666666666666
    Epoch 70 / 100, Training loss 62.28535842895508, Train acc 0.6633333333333333, Testing loss 0.970268726348877, Test acc 0.6266666666666667
    Epoch 71 / 100, Training loss 61.925902247428894, Train acc 0.662962962962963, Testing loss 0.9859861731529236, Test acc 0.64
    Epoch 72 / 100, Training loss 61.518729478120804, Train acc 0.6696296296296296, Testing loss 0.8854215145111084, Test acc 0.65
    Epoch 73 / 100, Training loss 61.065076768398285, Train acc 0.6696296296296296, Testing loss 0.8045321106910706, Test acc 0.66
    Epoch 74 / 100, Training loss 60.9826877117157, Train acc 0.6696296296296296, Testing loss 0.6010743975639343, Test acc 0.62
    Epoch 75 / 100, Training loss 60.828190833330154, Train acc 0.6714814814814815, Testing loss 1.127792477607727, Test acc 0.6533333333333333
    Epoch 76 / 100, Training loss 60.157768338918686, Train acc 0.6781481481481482, Testing loss 0.8585305213928223, Test acc 0.66
    Epoch 77 / 100, Training loss 59.997616946697235, Train acc 0.6759259259259259, Testing loss 0.7723979949951172, Test acc 0.6533333333333333
    Epoch 78 / 100, Training loss 59.67000788450241, Train acc 0.6785185185185185, Testing loss 0.6365461945533752, Test acc 0.6633333333333333
    Epoch 79 / 100, Training loss 59.36819651722908, Train acc 0.6759259259259259, Testing loss 0.9718449115753174, Test acc 0.6133333333333333
    Epoch 80 / 100, Training loss 58.985225826501846, Train acc 0.6837037037037037, Testing loss 0.7732844352722168, Test acc 0.6666666666666666
    Epoch 81 / 100, Training loss 58.64997532963753, Train acc 0.6807407407407408, Testing loss 0.7198848724365234, Test acc 0.6633333333333333
    Epoch 82 / 100, Training loss 58.309571266174316, Train acc 0.6874074074074074, Testing loss 0.6551756858825684, Test acc 0.62
    Epoch 83 / 100, Training loss 58.3891419172287, Train acc 0.6855555555555556, Testing loss 0.6093794107437134, Test acc 0.6266666666666667
    Epoch 84 / 100, Training loss 57.90271916985512, Train acc 0.6914814814814815, Testing loss 0.8765618205070496, Test acc 0.6333333333333333
    Epoch 85 / 100, Training loss 57.57546046376228, Train acc 0.6907407407407408, Testing loss 1.019386649131775, Test acc 0.6233333333333333
    Epoch 86 / 100, Training loss 57.38074314594269, Train acc 0.6833333333333333, Testing loss 0.6348428130149841, Test acc 0.6566666666666666
    Epoch 87 / 100, Training loss 56.92962437868118, Train acc 0.6937037037037037, Testing loss 0.7473028302192688, Test acc 0.6766666666666666
    Epoch 88 / 100, Training loss 56.58632490038872, Train acc 0.6940740740740741, Testing loss 0.5204653143882751, Test acc 0.67
    Epoch 89 / 100, Training loss 56.65665555000305, Train acc 0.6974074074074074, Testing loss 0.9957051873207092, Test acc 0.66
    Epoch 90 / 100, Training loss 56.24604853987694, Train acc 0.6937037037037037, Testing loss 0.9307649731636047, Test acc 0.6533333333333333
    Epoch 91 / 100, Training loss 56.273225992918015, Train acc 0.6977777777777778, Testing loss 0.6999312043190002, Test acc 0.65
    Epoch 92 / 100, Training loss 55.671839863061905, Train acc 0.6992592592592592, Testing loss 0.8036849498748779, Test acc 0.6566666666666666
    Epoch 93 / 100, Training loss 55.52847522497177, Train acc 0.702962962962963, Testing loss 0.7981866002082825, Test acc 0.67
    Epoch 94 / 100, Training loss 55.228808492422104, Train acc 0.7003703703703704, Testing loss 0.22902117669582367, Test acc 0.67
    Epoch 95 / 100, Training loss 54.87367296218872, Train acc 0.7074074074074074, Testing loss 0.8547374606132507, Test acc 0.6333333333333333
    Epoch 96 / 100, Training loss 54.62296414375305, Train acc 0.7162962962962963, Testing loss 0.8192398548126221, Test acc 0.6466666666666666
    Epoch 97 / 100, Training loss 54.30062720179558, Train acc 0.7066666666666667, Testing loss 0.8169623017311096, Test acc 0.6633333333333333
    Epoch 98 / 100, Training loss 54.35749989748001, Train acc 0.705925925925926, Testing loss 0.7754251956939697, Test acc 0.6766666666666666
    Epoch 99 / 100, Training loss 53.98323491215706, Train acc 0.7218518518518519, Testing loss 0.9443226456642151, Test acc 0.6666666666666666
    Epoch 100 / 100, Training loss 53.97128826379776, Train acc 0.7103703703703703, Testing loss 0.8664492964744568, Test acc 0.66


It is possible to vizualise in real-time the training of your model by using tensorboard. Once your model begins training you will see that a folder `runs` appeared. This folder contains all the metrics we computed in the training loop. You can vizualise the training by writing `tensorboard --logdir runs/` in the command line. Vizualizing the training in real time is extremely useful when you are experimenting with a new model: if your experiment is going wrong you can abort the training and save some time!

At the end of the training we have a model which have an accuracy of about 71% on the training set and 66% on the test set. Not too bad for a very simple CNN, we could add some complexity to the model by adding more layers for instance. The model also **does not seem to overfit too much** as the gain of accuracy between the training and testing dataset tend to be linear. However, it seems that around 70 epochs, while the training accuracy improves, the testing accuracy levels of. This could be solved by adding some complexity to the model such as adding dropout layers.

![](/post/2021-05-05-building-a-cnn-from-scratch-with-pytorch/tb_pic.PNG)

Now that we spent some time training the model we can save its weights, so we don't have to wait again as the training can literally take months! We can save the model parameters using the function `torch.save()`. This returns a dictionnary that maps each layer to its parameter tensor.


```python
torch.save(model.state_dict(), "models_weights/weights_simple_cnn.pt")
```

# Making predictions using the trained model

Now that we have a trained model we can use it to make some predictions. First of all we need to match the saved parameters to a new instance of our simple CNN. This means that we reinitiate a ConvNet and instead of retraining it, we match the `state_dict` weights to its neuron.


```python
# Instantiate a new ConvNet
model = ConvNet()
# Match the weigths of the state_dict to its neuron
model.load_state_dict(torch.load("models_weights/weights_simple_cnn.pt"))
```




    <All keys matched successfully>



Now we can define a small function that returns a batch of images (4 by default) and compute the predictions for each image:


```python
class_names = ["cats", "dogs", "pandas"]

def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader_test):
            inputs = inputs
            labels = labels

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                inputs_trans = inputs[j].squeeze().permute(2,1,0)
                plt.imshow(inputs_trans)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```


```python
visualize_model(model)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](/post/2021-05-05-building-a-cnn-from-scratch-with-pytorch/tutorial_49_1.png)
    


# End of this tutorial

And that's it for this tutorial! This was very hands-on and I hope you enjoyed it. We will see how we can use pre-trained models to drastically increase the accuracy of our classification problem!
