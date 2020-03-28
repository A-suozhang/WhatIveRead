import math

quan_length = 6

f = float(input())
f_1 = math.floor(f)
f_2 = f - f_1

print(bin(f_1)[2:],end = '')
print('.', end = '')
while(f_2 > 0):
    temp = f_2*2
    print(math.floor(f_2*2), end = '')
    f_2 = temp - math.floor(temp)
