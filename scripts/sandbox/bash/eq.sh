#!/bin/bash
echo "Enter A Number"
read a
b=100
b=$[b**3]
echo $b
if [ ${a} -eq ${b} ]; then
	echo "number equal"
else
	echo "number unequal"
fi
