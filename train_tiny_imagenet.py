import os
import sys
import keras
import argparse
import numpy as np
import tensorflow as tf
from data_utils import load_tiny_imagenet

from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint

# Suppress compiler warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Path to put computed activations/best epoch
train_path = os.path.join('work', 'training', 'tiny_imagenet')
if not os.path.isdir(train_path):
	os.makedirs(train_path)

def train_tiny_imagenet(hardware='cpu', batch_size=100, num_epochs=25, num_classes=10):
	# Load data
	x_train, y_train, x_val, y_val = process_images()
	
	if hardware == 'gpu':
		devices = ['/gpu:0']
	elif hardware == '2gpu':
		devices = ['/gpu:0', '/gpu:1']
	else:
		devices = ['/cpu:0']

	# Run on chosen processors
	for d in devices:
		with tf.device(d):
			model = Sequential()

			print(x_train.shape[1:])
			"""Block 1"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', 
					  input_shape=x_train.shape[1:]))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)
			
			"""Block 2"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			print(model.layers[-1].output_shape)

			"""Block 3"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			print(model.layers[-1].output_shape)

			"""Block 4"""
			model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Conv2D(512, (3, 3), strides=(1,1), padding='same'))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			print(model.layers[-1].output_shape)
						
			"""Block 5"""
			model.add(Flatten())
			print(model.layers[-1].output_shape)
			model.add(Dense(4096))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)
			
			"""Block Test"""
			model.add(Dense(1024))
			print(model.layers[-1].output_shape)
			model.add(BatchNormalization())
			print(model.layers[-1].output_shape)
			model.add(Activation('relu'))
			print(model.layers[-1].output_shape)

			"""Output Layer"""
			model.add(Dense(num_classes))
			print(model.layers[-1].output_shape)

			"""Loss Layer"""
			model.add(Activation('softmax'))
			print(model.layers[-1].output_shape)

			"""Optimizer"""
			model.compile(loss=losses.categorical_crossentropy, 
						  optimizer='adam', metrics=['accuracy'])

	# check model checkpointing callback which saves only the "best" network according to the 'best_criterion' optional argument (defaults to validation loss)
	model_checkpoint = ModelCheckpoint(train_path + 'best_weights_' + best_criterion + '.hdf5', monitor=best_criterion, save_best_only=True)

	if not data_augmentation:
		print('Not using data augmentation.')
		# Use the defined 'model_checkpoint' callback
		model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=num_epochs,
			  validation_data=(x_val, y_val),
			  shuffle=True, 
			  callbacks=[model_checkpoint])
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
						callbacks=[model_checkpoint])

def process_images(num_classes=10):
	# Path to tiny imagenet dataset
	#path = input('Enter the relative path to the directory containing the wnids/words files: ')
	path = os.path.join('tiny-imagenet-200')
	#path = os.path.join('tiny-imagenet-200', 'random', '0')
	print(path)
	# Generate data fields - test data has no labels so ignore it
	classes, x_train, y_train, x_val, y_val = load_tiny_imagenet(path, os.path.join('random', '0'), num_classes=num_classes, resize=True)
	# Get number of classes specified in order from [0, num_classes)
	print(classes)
	#print(x_train)
	print(x_train.shape)
	print(y_train.shape)

	"""
	if num_classes > 200:
		print('Set number of classes to maximum of 200\n')
		num_classes = 200
	elif num_classes != 200:
		train_indices = [index for index, label in enumerate(y_train) if label < num_classes]
		val_indices = [index for index, label in enumerate(y_val) if label < num_classes]
		x_train = x_train[train_indices]
		y_train = y_train[train_indices]
		x_val = x_val[val_indices]
		y_val = y_val[val_indices]
	"""

	# Format data to be the correct shape
	x_train = np.einsum('iljk->ijkl', x_train)
	x_val = np.einsum('iljk->ijkl', x_val)

	# Convert labels to one hot vectors
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	return x_train, y_train, x_val, y_val	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a convolutional neural network on the Tiny-Imagenet dataset.')
	parser.add_argument('--hardware', type=str, default='cpu', help='cpu, gpu, or 2gpu currently supported.')
	parser.add_argument('--batch_size', type=int, default=100, help='')
	parser.add_argument('--num_epochs', type=int, default=50, help='')
	parser.add_argument('--num_classes', type=int, default=10, help='')
	parser.add_argument('--data_augmentation', type=bool, default=False, help='')
	parser.add_argument('--best_criterion', type=str, default='val_loss', help='Criterion to consider when choosing the "best" model. Can also use \
																				"val_acc", "train_loss", or "train_acc" (and perhaps others?).')
	args = parser.parse_args()
	hardware, batch_size, num_epochs, num_classes, data_augmentation, best_criterion = args.hardware, args.batch_size, args.num_epochs, args.num_classes, args.data_augmentation, args.best_criterion

	# Possibly change num_classes to be a list of specific classes?
	train_tiny_imagenet(hardware, batch_size, num_epochs, num_classes)
