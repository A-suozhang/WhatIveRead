# Main Function

import re
import os
import matplotlib.pyplot as plt

num_file = 0
for s in os.listdir('./logs'):
    num_file = num_file + 1
    s0 = os.path.abspath('./logs/')+"/"+s
    print(s0)
    with open(s0,'r') as f:
        plt.figure()
        lines = f.readlines()
        line_num = len(lines)
        test_accs = []
        train_accs = []
        params = {}
        for i in lines:
            
            if ( "q_cfg :" in i):
                q_cfg_s = i
                params = re.findall(r'[[](.*?)[]]', q_cfg_s)
            elif ('q_cfg' in i):
                q_cfg_s = i
                _params = re.findall(r'[[](.*?)[]]', q_cfg_s)
                if(len(_params) == 7):
                    params = _params           

#             print(params)                 
                
            if (i[0:13] == "Test: [50/79]"):
                temp_s2 = i
                temp_indice = temp_s2.find("Acc@1")
                sliced_s2 = temp_s2[temp_indice+14:temp_indice+20]
                test_acc = float(sliced_s2.split(")")[0])
                test_accs.append(test_acc)
            if ("[350/391]" in i):
                temp_s = i
                temp_indice = temp_s.find("Acc@1")
#                 print(temp_s[temp_indice+14:temp_indice+20].split("(")[-1])
                train_acc = float(temp_s[temp_indice+14:temp_indice+20].split("(")[-1])
                train_accs.append(train_acc)

            # Especially For The 35 Pic
            if ("[2500/2503]" in i):
                temp_s = i
                temp_indice = temp_s.find("Acc@1")
                train_acc = float(temp_s[temp_indice+14:temp_indice+20].split("(")[-1])
                train_accs.append(train_acc)

            if ("[50/98]" in i):
                temp_s = i
                temp_indice = temp_s.find("Acc@1")
                test_acc = float(temp_s[temp_indice+14:temp_indice+20].split("(")[-1])
                test_accs.append(test_acc)
                
        
        if(len(params) == 3):
            plt.title("BITWIDTH: {} | Linear: {} | Level: {} |".format(params[0],params[1],params[2]))
        elif(len(params) == 5):
            plt.title("BITWIDTH: {} | Linear: {} | eRange: {} | \n Group: {} | Level: {}|".format(params[0],params[1],params[2],params[3],params[4]))
        elif(len(params) == 7):
            plt.title("BITWIDTH: {} | Linear: {} | Level: {} |".format(params[0],params[1],params[2]))
        else:
            plt.title("Unknow Q Config")
        plt.plot(train_accs)
        plt.plot(test_accs)
        plt.legend(['train','test'],loc='lower right')
        plt.savefig("./imgs/"+s+".jpg")
        
print(num_file)
