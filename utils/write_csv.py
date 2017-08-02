import os
import csv
import sys
sys.path.insert(0, os.path.join('networks'))
from train_tiny_lenet import train_tiny_imagenet

if __name__ == '__main__':
	start = int(input('Enter network number to start at: '))
	end = int(input('Enter network number to end at: '))
	
	data = [['SET', 'CLASSES', 'ACCURACY', 'BATCH_SIZE', 'EPOCHS', '# CLASSES', 'LEARNING RATE', 'WEIGHT DECAY', 'IMAGE SIZE']]
	for i in range(start, end+1):
		set_path = os.path.join('random', str(i))
		words10_path = os.path.join('sets', set_path, 'words10.txt')
		with open(words10_path) as f:
			classes = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			classes = [x.split('\t')[1].split(',')[0].strip() for x in classes]
		
		classes = str(classes).replace(',', '')
		model_path = os.path.join('work', 'training', 'tiny_imagenet', 'sets', set_path, 'best_weights_val_acc.hdf5')
		
		# ADD CODE HERE TO LOOP THROUGH EVERY HDF5 FILE IN SET DIRECTORY AND
		# EVALUATE THEM ALL TO WRITE TO CSV FILE

		# MAYBE ADD SOMETHING TO CHECK IF FILE WAS ALREADY WRITTEN TO TABLE
		# OR NOT

		# Get important network information for table
		
		# THIS IS WRONG, THESE WILL ALWAYS BE DEFAULT VALUES, NEED TO GET IT
		# FROM LOADED MODEL SOMEHOW
		accuracy, batch_size, num_epochs, num_classes, lr, decay, resize = train_tiny_imagenet(wnids=set_path, resize=True, load=model_path)
		accuracy = '%.2f%%' % float(accuracy)
		
		if resize:
			size = '32x32x3'
		else:
			size = '64x64x3'

		line = set_path + ',' + classes + ',' + accuracy + ',' + str(batch_size) + \
				',' + str(num_epochs) + ',' + str(num_classes) + ',' + str(lr) + \
				',' + str(decay) + ',' + size
		data.append(line.split(','))

	table_name = input('Enter table name to create or append to: ')
	mode = input("Enter 'w' for new table or 'a' to append to an existing \
				 table: ")
	outpath = os.path.join('results', table_name)

	with open(outpath, mode) as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for line in data:
			writer.writerow(line)

	csv_file.close()
