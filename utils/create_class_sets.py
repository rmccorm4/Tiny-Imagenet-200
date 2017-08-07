import os
from random import randint, sample

# Create words200.txt file
all_words_path = os.path.join('tiny-imagenet-200', 'words.txt')
all_wnids_path = os.path.join('tiny-imagenet-200', 'wnids.txt')

if not os.path.exists(os.path.join('sets')):
	os.mkdir(os.path.join('sets'))

os.system('cp ' + all_words_path + ' sets/')
os.system('cp ' + all_wnids_path + ' sets/wnids200.txt')
words200_path = os.path.join('sets', 'words200.txt')
os.system('grep -f ' + all_wnids_path + ' ' + all_words_path + ' > ' + words200_path)

# Define number of classes to put in a set
num_classes = int(input('Enter number of classes to put in a set: '))
# Define number of sets of classes to produce
sets = int(input('Enter the number of sets to generate: '))
start = int(input('Enter the number to start making sets from: '))

# Read in words200.txt file to generate sets of 10 classes
inpath = os.path.join('sets', 'words200.txt')
infile = open(inpath, 'r')

lines = []
for line in infile:
	lines.append(line)

infile.close()

# Produce sets of classes
for i in range(sets):
	temp = lines[:]
	outpath = os.path.join('sets', 'random', str(start+i))
	os.system('mkdir -p ' + outpath)
	words_path = os.path.join(outpath, 'words' + str(num_classes) + '.txt')
	words_file = open(words_path, 'w')
	
	wnids_path = os.path.join(outpath, 'wnids' + str(num_classes) + '.txt')
	wnids_file = open(wnids_path, 'w')

	# Get <num_classes> random classes
	words = sample(temp, num_classes)

	# Get class IDs separately
	for line in words:
		wnids_file.write(line.split('\t')[0])
		wnids_file.write('\n')
		words_file.write(line)

words_file.close()
wnids_file.close()
