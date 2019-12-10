#!/bin/bash

arr=(1 2 3 5 8 13)

total={#arr[*]}
echo "Num of Elements:"${total}

echo "Array Values:"
for var in ${arr[*]}; do
	printf "%d " ${var}
done

echo "Array Value with Index"
for key in ${!arr[*]}; do
	printf "%d: %s\n " ${key} ${arr[${key}]}
done

