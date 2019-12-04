print("INPUT THE BINARY STRING:\n")
s = input()
out = 0
s_1, s_2 = s.split('.')

for i in range(len(s_1)):
    out += pow(2,i)*int(s_1[-(i+1)])
for i in range(len(s_2)):
    out += pow(2,-(i+1))*int(s_2[i])

print(out)


