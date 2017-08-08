import os
import csv
import sys
sys.path.insert(0, os.path.join('networks'))
from train_tiny_lenet import train_tiny_imagenet

if __name__ == '__main__':
	#start = int(input('Enter network number to start at: '))
	#end = int(input('Enter network number to end at: '))
	start = int(sys.argv[1])
	end = int(sys.argv[2])
	
	# Title Row for Table
	data = [['SET', 'CLASSES', 'ACCURACY', 'NORMALIZE', 'BATCH_SIZE', 'EPOCHS', '# CLASSES', 'LEARNING RATE', 'WEIGHT DECAY', 'IMAGE SIZE']]
	
	# Get network info for every set
	for i in range(start, end+1):
		set_path = os.path.join('random', str(i))
		words10_path = os.path.join('sets', set_path, 'words10.txt')
		with open(words10_path) as f:
			classes = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			classes = [x.split('\t')[1].split(',')[0].strip() for x in classes]	
		classes = str(classes).replace(',', '')
		
		model_path = os.path.join('work', 'training', 'tiny_imagenet', 'sets', set_path)
		models = os.listdir(model_path)

		for model in models:
			if 'hdf5' in model:
				# Get network information from filename
				params = model.split('_')
				network = {'batch_size' : params[0], 'num_epochs' : params[1],
						   'num_classes' : params[2], 'lr' : params[3],
						   'decay' : params[4], 'resize' : params[5],
						   'normalize' : params[6]}
				
				accuracy = train_tiny_imagenet(wnids=set_path, 
											   resize=network['resize'], 
											   load=os.path.join(model_path, model),
											   normalize=network['normalize'])
				accuracy = '%.2f%%' % float(accuracy)

				if network['resize'].lower() == 'true':	
					size = '32x32x3'
				else:
					size = '64x64x3'

				line = str(set_path + ',' + classes + ',' + accuracy + ',' + \
					   network['normalize'] + ',' + network['batch_size'] + \
					   ',' + network['num_epochs'] + ',' + \
					   network['num_classes'] + ',' + network['lr'] + ',' + \
					   network['decay'] + ',' + size)
				data.append(line.split(','))
			
			else:
				continue

	# Write network info out to CSV file
	#table_name = input('Enter table name to create or append to: ')
	#mode = input("Enter 'w' for new table or 'a' to append to an existing" + \
	#			  " table: ")
	
	table_name = sys.argv[3]
	mode = sys.argv[4]
	
	outpath = os.path.join('results', table_name)

	with open(outpath, mode) as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for line in data:
			writer.writerow(line)

	csv_file.close()
