import os
import copy
import yaml
import numpy as np


with open("/home/foxfi/projects/prob_prune/pt_adv/exp_hyperparam/morphnet/res18_morphnet.yaml", "r") as rf:
   base_cfg = yaml.load(rf)

for lambda_ in [1e-10, 3e-10, 1e-9, 3e-9, 1e-8]:
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["morph_lambda"] = float(lambda_)
    with open("./lambda/lambda_{:.0e}.yaml".format(lambda_), "w") as wf:
        yaml.dump(cfg, wf)
