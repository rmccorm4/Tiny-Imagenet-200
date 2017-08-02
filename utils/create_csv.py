import os
import csv
import sys
sys.path.insert(0, 'networks/')
from train_loaded_tiny_lenet import train_tiny_imagenet

if __name__ == '__main__':
	start = int(input('Enter network number to start at: '))
	end = int(input('Enter network number to end at: '))
	
	data = [['SET', 'CLASSES', 'ACCURACY']]
	for i in range(start, end+1):
		set_path = 'random/' + str(i)
		words10_path = 'sets/' + set_path + '/words10.txt'
		with open(words10_path) as f:
			classes = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			classes = [x.split('\t')[1].split(',')[0].strip() for x in classes]
		
		classes = str(classes).replace(',', '')
		model_path = 'work/training/tiny_imagenet/sets/' + set_path + '/best_weights_val_loss.hdf5'
		accuracy = '%.2f%%' % float(train_tiny_imagenet(wnids=set_path, resize=True, load=model_path))
		
		line = set_path + ',' + classes + ',' + accuracy
		data.append(line.split(','))

	table_name = input('Enter table name to create or append to: ')
	mode = input("Enter 'w' for new table or 'a' to append to an existing
				 table: ")
	outpath = os.path.join('..', 'results', table_name)

	with open(outpath, mode) as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for line in data:
			writer.writerow(line)

	csv_file.close()
