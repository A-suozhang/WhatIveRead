a=1234
let "a+=10"
# echo ${a}

b="aa"
b=${b/aa/123}
# echo ${b}

c=""
let "c+=1"
# echo ${c}

d='123'
e=123

if [ $d != $e ]; then
       echo "Not Equal"
else
	if [ $b != $e ]; then 
		echo "Equal!"
	fi

fi       
	
