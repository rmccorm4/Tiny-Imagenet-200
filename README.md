# Tiny-Imagenet-200

This repository is my research code for exploration of Convolutional Neural
Networks, specifically on the Tiny-Imagenet-200 dataset. I plan to start
small with subsets of 10 classes to benchmark against CIFAR-10, then
eventually expand to larger and larger subsets, making my way up to all
200 classes to compare against [Stanford's CS231N results](https://tiny-imagenet.herokuapp.com/).

---
## Table of Contents

### Getting Started

### Setting Up Your Environment
* [Anaconda](https://www.continuum.io/anaconda-overview)
* [Pip](https://pypi.python.org/pypi/pip)

### Creating Class Sets

### Training Network

### Evaluating Trained Network

### Tweakable Parameters
---

## Getting Started

To use this code you will first need to download the dataset from
it's website: http://cs231n.stanford.edu/tiny-imagenet-200.zip

Alternatively, you can run the following command in your terminal
if you have `wget` installed to download it to your current directory:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

## Setting Up Your Environment

### Anaconda

Using Anaconda is a GREAT way to keep everything consistent
regardless of what machine you run your code on.

The following command will create an Anaconda virtual environment with the
modules listed in requirements.txt installed. This is very useful on clusters
where you don't necessarily have root priveliges. To be specific, I used
Anaconda/4.0.0 in these cases. Locally, any version should be fine.
```
conda create --name py3 python=3.5.2 --file requirements.txt
```

NOTE: Apparently if you create a model on one version of Keras, and load that
model in a different version of Keras, there is a good chance you'll get this
error:

```
ValueError: Optimizer weight shape (32,) not compatible with provided weight shape (5, 5, 3, 32)
```

To remedy this, I saved the conda environment with all of the versions that
were used to create the first 2000 models seen in utils/2000networks.csv in
utils/working_conda_env.yml. To recreate the environment with these exact
library verions, run this command:

```
conda env create -f utils/working_conda_env.yml
```

### Pip
Then you will need to install all required libraries by running the command:

```
pip install -r requirements.txt
```

## Creating class sets

To create sets of classes, just run `python produce_files.py` and enter
your desired number of sets, and number of classes per set.

Currently, I have number of classes set to 200 as default and image size
set to 64x64x3 as default.

To train this network, make whatever sets of classes you need using:

```
python utils/produce_files.py
```

and by following the prompts.

## Traning Network

Currently, everything is meant to be run from the highest level directory
of this repository. Paths could be incorrect if you run code from the directory
it is contained in, and this code would lose it's current generality.

To benchmark results against CIFAR-10, run the following command to resize
images from 64x64x3 to 32x32x3 and train the network on a 10-class subset on the
CIFAR LeNet architecture:

```
python networks/train_tiny_lenet.py --resize=True --num_classes=10
```

To choose which set of classes you train the network on when executing the
command, you can use the `wnids` optional argument to pass the relative
path to your chosen set of classes:

```
python networks/train_tiny_lenet.py --resize=True --num_classes=10 --wnids='random/0'
```

Otherwise, you will be prompted to input the path to a set of classes
when simply executing and default values for most parameters will be set:

```
python networks/train_tiny_lenet.py
```

## Evaluating Trained Network

To evaluate the accuracy of a network that's already been trained, you can use 
the `--load` optional argument as demonstrated below

```
python networks/train_tiny_lenet.py --resize=True --num_classes=10 --wnids='random/0' --load='work/training/tiny_imagenet/sets/random/0/best_weights_val_acc.hdf5'
```

## Tweakable Parameters

```
# String: Choice of whether to use 'cpu', 'gpu', '2gpu', Default='cpu'
--hardware

# Int: How many images to pass through the network at once, Default=100
--batch_size

# Int: How many times to run all of the data through th enetwork, Defualt=25
--num_epochs

# Int: Number of classes the network is being trained on
--num_classes

# Float: Adjustable hyperparameter, Default=0.001
--learning_rate

# Float: Adjustable hyperparameter, Default=0.00
--weight_decay

# Boolean: Whether to preprocess data in certain ways, Default=False
--data_augmentation

# String: Choice of 'train_acc', 'train_loss', 'val_acc', 'val_loss' to monitor
# for saving model checkpoints, Default='val_acc'
--best_criterion

# String: Path to set of classes to train on, Default=User_Input
--wnids

# Boolean: True=32x32, False=64x64, Default=False
--resize

# String: Path to saved model to evaluate accuracy of
--load

# Boolean: True=Normalize images by dividing each color channel by 255, Default=False
--normalize
```

## Notes To Self

TODO:
* Check if int values of images are correct as opposed to floats like in Matlab.
* Check if resized images look correct with imshow
* Networks are VERY dependent on weight initializations, so if they don't
get a good random start, the network accuracy will be random. Look into
how to set these values similarly to how matconvnet does it.
* Make choice of wnids path more dynamic
* Figure out how to check accuracy from saved model
	* Worst case run validation images through trained network again?

* Totally clean up network code and figure out how to extract important
network information from saved models or use callbacks/manually code it

IDEAS:
* Train a network to learn best classes to put together in a set?
* Without using a network, recursively train a network, get the best classes
and train on those, and repeat
* Can a Generative Adversarial Network be used to increase the number of 
training/validation images per class by generating realistic images?

ISSUES:
* Evaluating loaded model with validation data is not returning the proper 
accuracy, need to look into this...
