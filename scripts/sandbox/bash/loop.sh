#!/bin/bash

echo "Testing For Loop"
for i in 1 2 3 4 5; do
	printf "%d " ${i}
	python test${i}.py
done

echo "\nTesting While Loop"
cnt=0
while [ ${cnt} -le 10 ]; do 
	((cnt++))
	echo ${cnt}
done




