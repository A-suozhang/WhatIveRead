import os
import shutil

os.mkdir('./logs')

filelist = []
filelist_full = []
for home, dirs, files in os.walk(os.getcwd()):
    for filename in files:
        if (filename.split('.')[-1] == 'log'):
            filelist_full.append(os.path.join(home, filename))
            filelist.append(filename)

import shutil
k = 0
for i,j in zip(filelist, filelist_full):
    shutil.copyfile(j,"logs/"+str(k)+"_"+i)
    k = k+1

print("{} Logs Are Copied into './logs' Folder".format(k))
