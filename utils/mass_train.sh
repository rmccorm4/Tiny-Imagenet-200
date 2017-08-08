#!/bin/bash

cd '..'

for i in $(seq $1 $2)
do
	python networks/train_tiny_lenet.py --resize="true" --wnids="random/$i" --num_classes=10 --normalize="False"
	python networks/train_tiny_lenet.py --resize="true" --wnids="random/$i" --num_classes=10 --normalize="True"
done
