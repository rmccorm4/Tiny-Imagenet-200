For the sake of speed in cloning/pushing/pulling to this repository,
I've added the Tiny-Imagenet-200 dataset to the gitignore.

To use this code you will first need to download the dataset from
it's website: http://cs231n.stanford.edu/tiny-imagenet-200.zip

Alternatively, you can run the following command in your terminal
if you have `wget` installed to download it to your current directory:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

To create sets of classes, just run `python produce_files.py` and enter
your desired number of sets, and number of classes per set.

Currently, I have number of classes set to 10 as default and image size
set to 32x32x3 as default in order to train networks on the CIFAR-LeNet
architecture, as seen in `train_tiny_lenet.py`

To train this network, make whatever sets of classes you need using
`produce_files.py` or use the hand-picked sets I have provided.

Specify the path to the sets you chose inside of the training script and
run it with `python train_tiny_lenet.py`.

TODO:
* Check if int values of images are correct as opposed to floats like in Matlab.
* Check if resized images look correct with imshow
* Networks are VERY dependent on weight initializations, so if they don't
get a good random start, the network accuracy will be random. Look into
how to set these values similarly to how matconvnet does it.
* Make choice of wnids path more dynamic
