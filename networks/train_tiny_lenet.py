import os
import sys
import keras
import pickle
import argparse
import numpy as np
import tensorflow as tf
from data_utils import load_tiny_imagenet

from keras import losses
from keras import optimizers
from keras import initializers
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

# Suppress compiler warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Path to put computed activations/best epoch
train_path = os.path.join('work', 'training', 'tiny_imagenet')
if not os.path.isdir(train_path):
	os.makedirs(train_path)

# Train network
def train_tiny_imagenet(hardware='cpu', batch_size=100, num_epochs=25, 
						num_classes=10, lr=0.001, decay=0.00, wnids='', 
						resize='False', load='', normalize='False'):
	# Load data
	x_train, y_train, x_val, y_val, wnids_path = process_images(wnids, resize, num_classes, normalize)
	
	# Choose seleted hardware, default to CPU
	if hardware == 'gpu':
		devices = ['/gpu:0']
	elif hardware == '2gpu':
		devices = ['/gpu:0', '/gpu:1']
	else:
		devices = ['/cpu:0']
	
	# Run on chosen processors
	for d in devices:
		with tf.device(d):
			# Load saved model and check its accuracy if optional arg passed
			if load != '':
				model = load_model(load)
				# Run validation set through loaded network
				score = model.evaluate(x_val, y_val, batch_size=batch_size)
				#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
				return str(score[1]*100)

			# Otherwise train new network
			else:
				model = Sequential()

				"""Block 1"""
				model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same', 
						  kernel_initializer=initializers.random_uniform(minval=-0.01, maxval=0.01),
						  bias_initializer='zeros',
						  input_shape=x_train.shape[1:]))
				model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
				model.add(Activation('relu'))
				
				"""Block 2"""
				model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same',
						  kernel_initializer=initializers.random_uniform(minval=-0.05, maxval=0.05),
						  bias_initializer='zeros'))
				model.add(Activation('relu'))
				model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
				
				"""Block 3"""
				model.add(Conv2D(64, (5, 5), strides=(1,1), padding='same',
						  kernel_initializer=initializers.random_uniform(minval=-0.05, maxval=0.05),
						  bias_initializer='zeros'))
				model.add(Activation('relu'))
				model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

				"""Fully Connected Layer and ReLU"""
				model.add(Flatten())
				model.add(Activation('relu'))
				
				"""Output Layer"""
				model.add(Dense(num_classes,
						  kernel_initializer=initializers.random_uniform(minval=-0.05, maxval=0.05),
						  bias_initializer='zeros'))

				"""Loss Layer"""
				model.add(Activation('softmax'))

				"""Optimizer"""
				model.compile(loss=losses.categorical_crossentropy, 
							  optimizer=optimizers.adam(lr=lr, decay=decay), 
							  metrics=['accuracy'])
	
				# check model checkpointing callback which saves only the "best" network according to the 'criteria' optional argument
				sets_index = wnids_path.find('sets')
				outpath = os.path.join(train_path, wnids_path[sets_index:])
				# Naming file by hyperparameters
				resize = resize.title()
				normalize = normalize.title()
				prefix = '%d_%d_%d_%.5f_%.2f_%s_%s_best_%s_' % (batch_size, num_epochs, num_classes, lr, decay, resize, normalize, criteria)
				model_outfile = os.path.join(outpath, prefix + 'model.hdf5')
				csv_outfile = os.path.join(outpath, prefix + 'log.csv')
				if not os.path.exists(outpath):
					os.makedirs(outpath)

				# Save network state from the best <criteria> of all epochs ran
				model_checkpoint = ModelCheckpoint(model_outfile, monitor=criteria, save_best_only=True)
				# Log information from each epoch to a csv file
				logger = CSVLogger(csv_outfile)
				callback_list = [model_checkpoint, logger]

				if not data_augmentation:
					print('Not using data augmentation.')
					# Use the defined 'model_checkpoint' callback
					model.fit(x_train, y_train,
						  batch_size=batch_size,
						  epochs=num_epochs,
						  validation_data=(x_val, y_val),
						  shuffle=True, 
						  callbacks=callback_list)
				else:
					print('Using real-time data augmentation.')
					# This will do preprocessing and realtime data augmentation:
					datagen = ImageDataGenerator(
						featurewise_center=False,  # set input mean to 0 over the dataset
						samplewise_center=False,  # set each sample mean to 0
						featurewise_std_normalization=False,  # divide inputs by std of the dataset
						samplewise_std_normalization=False,  # divide each input by its std
						zca_whitening=False,  # apply ZCA whitening
						rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
						width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
						height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
						horizontal_flip=True,  # randomly flip images
						vertical_flip=False)  # randomly flip images

					# Compute quantities required for feature-wise normalization
					# (std, mean, and principal components if ZCA whitening is applied).
					datagen.fit(x_train)

					# Fit the model on the batches generated by datagen.flow().
					# Use the defined 'model_checkpoint' callback
					model.fit_generator(datagen.flow(x_train, y_train,
									batch_size=batch_size),
									steps_per_epoch=x_train.shape[0] // batch_size,
									epochs=num_epochs,
									validation_data=(x_val, y_val),
									callbacks=callbacks)
				
				return 'New network trained!'

def process_images(wnids_path='', resize='False', num_classes=200, normalize='False'):
	# Path to tiny imagenet dataset
	if wnids_path == '':
		wnids_path = input('Enter the relative path to the directory containing the wnids/words files from sets/: ')
	wnids_path = os.path.join('..', 'sets', wnids_path)
	
	# Generate data fields - test data has no labels so ignore it
	classes, x_train, y_train, x_val, y_val = load_tiny_imagenet(os.path.join('tiny-imagenet-200'), wnids_path, num_classes=num_classes, resize=resize)
	
	# Format data to be the correct shape
	x_train = np.einsum('iljk->ijkl', x_train)
	x_val = np.einsum('iljk->ijkl', x_val)

	if normalize.lower() == 'true':
		x_train /= 255
		x_val /= 255

	# Convert labels to one hot vectors
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	return x_train, y_train, x_val, y_val, wnids_path

# If running this file standalone
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a convolutional neural network on the Tiny-Imagenet dataset.')
	parser.add_argument('--hardware', type=str, default='cpu', help='cpu, gpu, or 2gpu currently supported.')
	parser.add_argument('--batch_size', type=int, default=100, help='')
	parser.add_argument('--num_epochs', type=int, default=25, help='')
	parser.add_argument('--num_classes', type=int, default=200, help='')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='')
	parser.add_argument('--weight_decay', type=float, default=0.00, help='')
	parser.add_argument('--data_augmentation', type=bool, default=False, help='')
	parser.add_argument('--criteria', type=str, default='val_acc', help='Criteria to consider when choosing the "best" model. Can also use "val_loss", "train_loss", or "train_acc".')
	parser.add_argument('--wnids', type=str, default='', help='Relative path to wnids file to train on.')
	parser.add_argument('--resize', type=str, default='False', help='False = 64x64 images, True=32x32 images')
	parser.add_argument('--load', type=str, default='', help='Path to saved model to load and evaluate.')
	parser.add_argument('--normalize', type=str, default='False', help='Path to saved model to load and evaluate.')
	
	args = parser.parse_args()
	hardware, batch_size, num_epochs, num_classes, lr, decay, data_augmentation, criteria, wnids, resize, load, normalize = args.hardware, args.batch_size, args.num_epochs, args.num_classes, args.learning_rate, args.weight_decay, args.data_augmentation, args.criteria, args.wnids, args.resize, args.load, args.normalize

	# Possibly change num_classes to be a list of specific classes?
	train_tiny_imagenet(hardware, batch_size, num_epochs, num_classes, lr, decay, wnids, resize, load, normalize)
