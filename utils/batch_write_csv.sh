#!/bin/bash

cd ..

# $1 = Start
# $2 = End
# $3 = Step size

for i in $(seq $1 $3 $2)
do
	python utils/write_csv.py $i $[i-1+$3] 2000network.csv a
done	
