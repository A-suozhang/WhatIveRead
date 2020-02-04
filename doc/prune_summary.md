# Cifar10

## ResNet18

### Morph

|Lambda|Truncate|Acc|FLOPs|
|--|--|--|--|
|1e-10|1e-2|94.2|50%|
|1e-9|1e-2|93.4|23.4%|
|1e-9|1e-3|93.6|32.9%|
|1e-9|2e-2|93.4|18.6%|
|1e-9|3e-2|92.8|14.6%|
|3e-9|1e-2|91.83|8.182%|
|1e-8|1e-2|87.05|3.239%|

### Lasso

|Lambda|Truncate|Acc|FLOPs|
|--|--|--|--|
|3e-4|1e-3|/|100%|
|1e-3|1e-3|94.4|80%|
|3e-3|1e-3|93.07|50%|
|5e-3|1e-3|91.07|25%|
|1e-2|1e-3|86.2|10%|
|3e-2|1e-3|/|/|

### Plan1

|Beta2 Schedule|Acc|Sparsity|Record|
|--|--|--|--| 
|1.1(50)|93.9|50%|1/|

## Vgg16

### Lasso
|Lambda|Truncate|Acc|FLOPs|
|--|--|--|--|
|1e-4|1e-3|92.8|98%|
|1e-3|1e-3|93.08|55%|
|3e-3|1e-3|91.6|25%|  


## TODO
* ~~trunc 3e-2的morph - 12.5%Sparsity~~
* ~~5e-3的lass~~o
* 继续调整plan1
* lambda可以很好的控制pruneratio
  * 优点控制的太好了...需要调好几个数量级
* beta2-grow建议调快点
  * 目前的2还比较好用
* 现在plan1的behaviour 
	* 跑到50个epoch左右会报错overflow但是能够继续运行下去
	* 一开始Sparsity会迅速下降之后就不变了
	* 后续慢慢提高精度
* 我们这个想不相当于finetune了已经，是不是也应该给它们finetune？
  * 也不尽然
* 不太好的现象是我们跑出来的东西各个PC的还是比较平均的...
  * 有没有很好的学到层间tradeoff呢...
