#!/bin/bash

echo "Enter a string"
read text

if (( ${text} == "fuck" )); then 
	echo "Language!"
else
	echo "Got It."
fi
