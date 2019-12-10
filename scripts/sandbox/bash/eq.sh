#!/bin/bash
echo "Enter A Number"
read a
b=100
if [[ ${a} -eq ${b} ]]; then
	echo "number equal"
else
	echo "number unequal"
fi
