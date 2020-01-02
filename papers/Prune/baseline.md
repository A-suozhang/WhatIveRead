# Baseline For Network Prunning

> Here Concludes Some Baseline Reported In Recent Papers

## Variational Netowrk Prunning

* LR Decay, SGD(Momentum 0.9; Weight Decay 1e-4)
* DataAug : Random Flip & Clip
* Trained From Scratch
	* Threshold Set
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223205850.png)
* VGG(13Conv+3fc)

### Resluts

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223210558.png) 
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223210722.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223211119.png)
	* [29] - ThiNet
	* [27] - Learning Efficient CNN through Network Slimming


## Self-Adaptive Prunning

* Additional Computation of SPM is Really Small

### Results

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223211853.png)
	* Cifar10
	* Gao2019 - Dynamic Channel Prunning: Feature Boostig and Suppression
	* Hua2018 - ChannelGaringNetwork
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223212325.png)
	* Cifar100


## Transferable NAS

### Results

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223213201.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223213243.png)
	* FPGM - Pruning filter via geometric median for deep convolutional neural networks acceleration (CVPR)
	* SFP - Soft filter pruning for accelerating deep convolutional neural networks(IJCAI 2018)
	* LCCL - More is Less (CVPR2017)
	* AutoSlim - AutoSlim: Towards One-Shot Architecture Search for Channel Numbers

## NetAdpat

* ADC - Automated deep compression and acceleration with reinforcement learning. arXiv preprint arXiv:1802.03494 (2018)
* MorphNet - Fast & simple resource-constrained structure learning of deep net- works. In:(CVPR)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223221736.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223221758.png)



## AutoPrune

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223220308.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223220313.png)

## AutoCompress

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223231636.png) 
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223231620.png)
	* Also Could be applied in Non-Structured Prunning
	* In ImageNet Experiment, Using TOP5 Acc Loss, without Showing Top5 Acc. (~)

## ECC

* Focused On Energy, No Further Results.



## Structured Probablistic

> Weird Baseline` 

## BAR 

* Couldn't Get it Yet, Maybe Latter Add it
	

## MaskConvNet

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223222657.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223222723.png)

## AMC

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223225252.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223225323.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223225409.png)
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191223225443.png)

## BAR
* Given Compression Ratio / Test Acc.
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191224145408.png)


# Final Table

> The End2End Method, Whether Finetuning

## The Prune Table

|Method | Network | Dataset | Acc | Sparsity | Notes |
|--|--|--|--|--|--|
|Baseline|Vgg16|Cifar10|93.66%|0%|(May Differ From Methods)|
|Baseline|Res18|Cifar10|94.6%|0%||(May Differ From Methods)|
|Variational. |VGG16|Cifar10|93.18%|62 % Removed|(Saved 70% Params & 40% Computation),The Sparisty is Channel Sparsity|
|Variational. |DenseNet|Cifar10|93.16%|60%|(Saved 60& Param & 45 % Computation & Large MemoryFootprint)|
|Variational. |Res20 |Cifar10|91.66%|38%|(Saved 20% Params & 16.47% Computation)|
|SelfAdaptive(SPM)|VGG|Cifar10|93.82%|66.4|(/)| 
|TransNAS|Res20|Cifar10|92.88%|45%|(45% FLOPs Pruned)|
|TransNAS|Res32|Cifar10|93.16%|41.5%|(41.5% FLOPs Pruned)|
|TransNAS|Res56|Cifar10|92.69%|52.7%|(52.7% FLOPs Pruned)|
|AutoPrune|VGG-like|Cifar10|92.2%|75CR(Fine-Grained)|(92.4%-92.2%)|
|MaskConvNet|VGG16|Cifar10|93.4%|88.53%|(40% FLOPs & 88.53% Params)|
|MaskConvNet|Res56|Cifar10|92.25%|31.87%|(252.27% FLOPs & 31.87% Sparsity)|
|AutoCompress|VGG16|Cifar10|93.21&|50x|50x Param Ratio; 8.8x FLOPsRatio|
|AutoCompress|Res18|Cifar10|93.8|54.2x|54.2 ParamRatio; 12.2x FLPOsRatio(Reported Baseline as 93.9%)|
|Baseline|AlexNet|ImageNet|56.8~57.2%|0%|(Range From Methods)|
|Baseline|Res18|ImageNet|69.8%|0%|(May Differ From Methods)||
|Baseline|Res50|ImageNet|75.1%|0%|(May Differ From Methods)||
|Variational.| Res50|ImageNet|75.2%|40%|()|
|TransNAS|Res18|ImageNet|69.15%|33.3%|(FLOPs Pruned)|
|TransNAS|Res50|ImageNet|76.2%|43.5%|(FLOPs Pruned)|
|MaskConvNet|Res34|ImageNet|72.56|10.75%|(10.75 FLOPs & 19 Params)|

## The Prune-Under-Budget Charts

> Often in form of (Acc. Under Certain arbitary PruneRatio((PR))

* [Adam-ADMM]() 
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231181647.png)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231181715.png)

* [MorphNet]()
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231182023.png)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231182042.png)

* [AMC]()
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231182132.png)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231182215.png)
		* 它居然有V2的实验结果？

* [BAR]()
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191231192035.png)


* [NetAdapt]()
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200101184632.png)

* [LeGR](https://arxiv.org/pdf/1904.12368.pdf)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200101213411.png)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200101213446.png)
