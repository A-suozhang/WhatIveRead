import logging
import os

logger = logging.getLogger('mylogger') 
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('test.log') 
fh.setLevel(logging.DEBUG) 

formatter = logging.Formatter("")
fh.setFormatter(formatter) 
logger.addHandler(fh) 

for s in os.listdir('./logs'):
    s0 = os.path.abspath('./logs/')+"/"+s
    print(s0)
    with open(s0,'r') as f:
        lines = f.readlines()
        line_num = len(lines)
    for i in lines:
        if ( "quantize_cfg :" in i):
            temp_s = i
    logger.info(temp_s)
