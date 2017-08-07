#!/bin/bash
cd ..
python networks/train_tiny_lenet.py --resize=True --wnids="random/250" --num_classes=10 --normalize False
