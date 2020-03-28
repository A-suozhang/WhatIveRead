from pytorch_adv.models import get_model
import torch
import torch.nn as nn
import ipdb
from IPython import embed

# MODEL_NAME = "cifar10_resnet20"
MODEL_NAME = "resnet18"

model = get_model(MODEL_NAME)()
print(model)


# x = torch.randn([1,3,32,32])
x = torch.randn([1,3,224,224])
out = model(x)

print("--------------------{}---------------------------".format(MODEL_NAME))
print("Conv Ops Forward {:.3E}".format(model.conv_ops_f))
print("Conv Ops Backward {:.3E}".format(model.conv_ops_b))
print("BN Ops {:.3E}".format(model.bn_ops))
print("Element-Wise Ops Forward {:.3E}".format(model.ew_add_ops_f))
print("Element-Wise Ops Backward {:.3E}".format(model.ew_add_ops_b))
print("Conv Params {:.3E}".format(model.conv_params))
print("BN Params {:.3E}".format(model.bn_params))

print("----------------------------------")

print("--------- Network Compoenents -------------")
for i,j in model.named_modules():
    if isinstance(j, nn.Conv2d):
        print("Conv: "+i)
    elif isinstance(j, nn.BatchNorm2d):
        print("BN: "+i)
    elif isinstance(j, nn.Sequential) and len(list(j.named_modules()))<2:
        print("EleAdd: "+i)
print("----------------------------------")


