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
