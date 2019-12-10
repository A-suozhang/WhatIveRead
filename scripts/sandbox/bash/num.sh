#!/bin/bash

num=120
let "num+=20"
((a=${num}-20))

echo ${num}  ${a}

function add(){
	local n=4
	local m=6
	((num=n+m))
	echo ${num}
}

n=100
add
echo "The num is:" ${num}

myArr=(1 2 3 5 8)
total=${#myArr[*]}
echo "Num Of Elements" ${total}

echo "Arr Values:"
for val in ${myArr[*]}; do
	printf "%d\n"  ${val}
	printf "%f\n"  ${val}
done
