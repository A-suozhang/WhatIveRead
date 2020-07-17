import os
import ipdb
import numpy as np
import re

# f_name_0 = "./get_epoch.log"
f_name_1 = "./train_0.log"
f_name_2 = "./train.log"


# 0 being the sparsity allocation
# with open(f_name_0, "r") as f:
#     lines_0 = f.readlines()

# 1 being the real acc & loss
# Need to change it at last to fit
with open(f_name_1, "r") as f:
    lines_1 = f.readlines()

with open(f_name_2, "r") as f:
    lines_2 = f.readlines()

true_tests = []
train = []
save = []
for line in lines_1:
    x = re.search("^Test", line)
    y = re.search("^Loss", line)
    z = re.search("^Sav", line)
    if x is not None:
        true_tests.append(line)
    if y is not None:
        train.append(line)
    if z is not None:
        save.append(line)

true_tests_2 = []
for line in lines_2:
    x = re.search("^Test", line)
    if x is not None:
        true_tests_2.append(line)

tests = np.array(true_tests)
# tests[:40] = true_tests[:40]

import ipdb; ipdb.set_trace()

cnt_x = 0
cnt_y = 0
for idx, line in enumerate(lines_2):
    x = re.search("^Test", line)
    y = re.search("^Loss", line)
    z = re.search("^Sav", line)
    if x is not None:
        if cnt_x > 50:
            lines_2[idx] = tests[cnt_x]
        cnt_x = cnt_x+1
    if y is not None:
        if cnt_y > 50:
            lines_2[idx] = train[cnt_y]
        cnt_y = cnt_y + 1
    if z is not None:
        lines_2[idx] = ""

with open("./new_log.log","w") as f:
    for line in lines_2:
        f.write(line)




