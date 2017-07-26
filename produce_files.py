import os
from random import randint

# Create words200.txt file
all_words_path = os.path.join('tiny-imagenet-200', 'words.txt')
all_wnids_path = os.path.join('tiny-imagenet-200', 'wnids.txt')
words200_path = os.path.join('tiny-imagenet-200', 'words200.txt')
os.system('grep -f ' + all_wnids_path + ' ' + all_words_path + ' > ' + words200_path)

# Define number of classes to put in a set
num_classes = int(input('Enter number of classes to put in a set: '))
# Define number of sets of classes to produce
sets = int(input('Enter the number of sets to generate: '))

# Read in words200.txt file to generate sets of 10 classes
inpath = os.path.join('tiny-imagenet-200', 'words200.txt')
infile = open(inpath, 'r')

lines = []
for line in infile:
	lines.append(line)

infile.close()

# Produce sets of classes
for i in range(sets):
	temp = lines[:]
	outpath = os.path.join('tiny-imagenet-200', 'random', str(i))
	os.system('mkdir -p ' + outpath)
	words_path = os.path.join(outpath, 'words' + str(num_classes) + '.txt')
	words_file = open(words_path, 'w')
	
	wnids_path = os.path.join(outpath, 'wnids' + str(num_classes) + '.txt')
	wnids_file = open(wnids_path, 'w')

	words = []
	# Get <num_classes> random classes
	for k in range(num_classes):
		choice = randint(0, len(temp)-1)
		# Pick class and remove it from list for no duplicates
		words.append(temp.pop(choice))

	# Get class IDs separately
	for line in words:
		wnids_file.write(line.split('\t')[0])
		wnids_file.write('\n')
		words_file.write(line)

words_file.close()
wnids_file.close()
