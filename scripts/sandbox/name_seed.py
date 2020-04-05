import os
import re
import numpy as np
import ipdb


for home, dirs, files in os.walk("./"):
    for file in files:
        filepath = os.path.join(home,file)
        print(filepath)


        if "log" in filepath:

            seed = re.search("search_(?P<seed>.*)\.log",filepath).group("seed")

            with open(filepath, "r") as f:
                lines = f.readlines()

            for idx, l in enumerate(lines):
                if re.search("seed",l) is not None:
                    if re.search("seed:\ (?P<seed>.*)\.",l) is not None:
                        seed2replace = re.search("seed:\ (?P<seed>.*)\.",l).group("seed")
                        lines[idx] = l.replace(seed2replace, str(int(seed)*1000))
                        print(lines[idx])
                    if re.search("seed\ (?P<seed>.+?)\ ",l) is not None:
                        seed2replace = re.search("seed\ (?P<seed>.*)\ ",l).group("seed")
                        lines[idx] = l.replace(seed2replace, str(int(seed)*1000))
                        print(lines[idx])

            with open(filepath, "w") as f:
                for l in lines:
                    f.write(l)


  
