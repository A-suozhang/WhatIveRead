# [Self-Adpative Network Prunning](https://arxiv.org/pdf/1910.08906.pdf)


# [Pruuning From Scratch](https://arxiv.org/pdf/1909.12579.pdf)


# [Adversal Neural Prunning](https://arxiv.org/pdf/1908.04355.pdf)

# [Rethinking The Value Of Network Prunning](https://arxiv.org/pdf/1810.05270v2.pdf)

# [Network Prunning via Transformable Architecture Search](https://arxiv.org/pdf/1905.09717v5.pdf)


---


# [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/pdf/1711.06798.pdf)

* Simply Applying Regularization

# [Designing energy- efficient convolutional neural networks using energy-aware pruning.](https://arxiv.org/abs/1611.05128)
* CVPR 2017
* Use Energy-Consumption to guide prunning process, use parameter extrapolated from actual hardware
	* 只是优先根据能耗大的先Prune而已...
	* 指出原先的方法都直接Target at减少模型压缩
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217104405.png)o
	* 依据能耗确定Prune Ratio
	* 先按照Magnitude剪，再Restore一部分
	* Locally Finetune，不做Backprop，基于Least Mean Square
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217104455.png)
		* 前几步都不会实际改变Weight的值
		* 看起来这篇文章也是专注于Minimize Output（Feature Map） Error，而不是Filter（weight） Error
* Layer-by-Layer Prune, first fintune according to least-squre error, then globally finetune


# [Netadapt: Platform-aware neural network adaptation for mobile applications.](https://arxiv.org/abs/1804.03230)
* ECCV 2018
* 也是认为MAC数目这类indirect的Metric不能直接完成任务，本文通过empirical measurement（？）来描述direct metric
	* A Framework optimize a pretrained Network to meet the resource budget
	* Automated Constraint Optimization Algorithm
	* Easy to Interpret
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217105524.png)
	* 将大问题抽象为一个Non-Convex constraint Problem
		*  ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217105642.png)
		* Res是DircectMetric，需要小于限制的Budget / Acc就是直接为Acc
		* 分解为A Stream of子问题
			* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217105859.png)
* [others Implemention](https://github.com/madoibito80/NetAdapt)


# [Layer- compensated pruning for resource-constrained convolutional neural networks](https://arxiv.org/abs/1810.00518)
* 将原先的1）How Many Filter To Prune 2) Which Filter To Prune with Given Prunning Ratio Merge成一个 Global Filter Ranking Problem
	*　原本的问题也就是LayerScheduling(确定每层需要剪多少的Ratio)加上Local Ranking
		* 而经常LayerScheduling相对较为依赖Expert Estimation或者是Empirical的方法
		* 作者认为这样不可靠，改变了Paleto Frontier
* Global Ranking的方法之前也有
	* 用启发式的方法做Global排序，贪婪地每次剪枝
		* Taylor Approxiamtion
		* Fisher Information
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191217113835.png)
* 还是偏启发式的一种方式...

# [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/abs/1903.03777)
* 侧重点在NAS,其实是搜索空间剪枝...我放它进来的时候还以为它是利用NAS做剪枝，其实并不是
* 也没有实际上硬件测Latency(他们也没有条件)，也是类似推算或者是打表的方式
* 从Wide和Depth两个角度建模模块相关关系

---
> 2019-12-18



[Structured Probabilistic Pruning for Convolutional Neural Network Acceleration](https://arxiv.org/pdf/1709.06994.pdf)
* BMVC 2017
* 不是确定性的剪枝，而是以一定的概率进行剪枝，提出了一个所谓的SPP
* 建模
	* 用L1 Norm的Rank来指导delta_p(Prune Ratio
	* 去构建一个函数 f（Rank）它应该有这样的性质：
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191218165037.png)
	* 最后的函数
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191218165957.png)
* 作者提到了使用Rank而不是直接用L1范数（虽然后者更直接更合理）但是意思是说用后者不work，认为Rank是加了一个软化
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191218170404.png)

[Accelerate CNN via Recursive Bayesian Pruning](https://arxiv.org/abs/1812.00353)
* ICCV 2019

[The Lottery Hypothesis: Finding Sparse Trainable Neural Network](https://arxiv.org/pdf/1803.03635.pdf)
* MIT ICLR2019 Best 
* Net Contains SubNetwork Their Initialization made them traninig good
	* The Old Prune Technique is actually finding them 
	* The Key is The Initialization
* The Problem : Identify The Lottery
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191220105629.png)
	* 上面流程是One-Shot的，通过N个Step，每个Step压最终ratio的1/N来iterative Pruning可以得到更好的结果
* Unstructed Prunning (The Final Results Are Sparse)
* High LR reuquires Warmup
* Think That SGD - Trains Well-Initialized Weights
	* Cause Dense Model Are easier to train, so prune over a over-parameterized model help find the GOOD WEIGHT INITIALIZATION of a Sparse Model



# 2019-12-20 ICLR-2020

> 总体看起来剪枝的文章好像不是特别吃香,这领域确实已经发展到了我们的未知领域


## Oral

* [Comparing Fine-tuning and Rewinding in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB)

## Spotlight

* [Compression based bound for non-compressed network: unified generalization error analysis of large compressible deep neural network](https://openreview.net/forum?id=ByeGzlrKwH)

* [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://openreview.net/forum?id=HJeTo2VFwH)

* [Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks](https://openreview.net/forum?id=BJxsrgStvr)

## Poster

* [Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP ](https://openreview.net/forum?id=S1xnXRVFwH)

* [Data-Independent Neural Pruning via Coresets](https://openreview.net/forum?id=H1gmHaEKwB)

* [One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation ](https://openreview.net/forum?id=r1e9GCNKvH)

* [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://openreview.net/forum?id=ryl3ygHYDB)

* [Dynamic Model Pruning with Feedback ](https://openreview.net/forum?id=SJem8lSFwB)

* [Provable Filter Pruning for Efficient Neural Networks ](https://openreview.net/forum?id=BJxkOlSYDH)
	* 这篇633的文章最后居然中了...

## Rejected

> 因为是rej所以只挑选了一些和我们比较相关或者比较有意思的文章拿出来看一下了

> 一些压BERT或者是图的文章,以及一些看上去就很神奇以及和我们没啥关系的文章省略了

* [Tensorized Embedding Layers for Efficient Model Compression ](https://openreview.net/forum?id=S1e4Q6EtDH)

* [Decoupling Weight Regularization from Batch Size for Model Compression ](https://openreview.net/forum?id=BJlaG0VFDH)

* [Differentiable Architecture Compression](https://openreview.net/forum?id=HJgkj0NFwr)

* [Evaluating Lossy Compression Rates of Deep Generative Models ](https://openreview.net/forum?id=ryga2CNKDH)

* [Teacher-Student Compression with Generative Adversarial Networks](https://openreview.net/forum?id=S1xipR4FPB)

* [Self-Supervised GAN Compression](https://openreview.net/forum?id=Skl8EkSFDr)

* [CAT: Compression-Aware Training for bandwidth reduction](https://openreview.net/forum?id=HkxCcJHtPr)
	* 下载下来他们的文章看了一眼,有格式问题+内容感觉没啥...
	* 这也给666...

* [FALCON: Fast and Lightweight Convolution for Compressing and Accelerating CNN](https://openreview.net/forum?id=BylXi3NKvS)

* [One-Shot Neural Architecture Search via Compressive Sensing](https://openreview.net/forum?id=B1lsXREYvr)

* [Domain-Invariant Representations: A Look on Compression and Weights ](https://openreview.net/forum?id=B1xGxgSYvH)
	* 完了他们这个也开始做了...

* [Progressive Compressed Records: Taking a Byte Out of Deep Learning Data ](https://openreview.net/forum?id=S1e0ZlHYDB)
	* 这个idea都9102年了还有人做吗?

* [Online Learned Continual Compression with Stacked Quantization Modules](https://openreview.net/forum?id=S1xHfxHtPr)
	* Online Compression

* [Atomic Compression Networks](https://openreview.net/forum?id=S1xO4xHFvB)

* [INTERPRETING CNN COMPRESSION USING INFORMATION BOTTLENECK](https://openreview.net/forum?id=S1gLC3VtPB)
	* 果然有人做,但是没中

* [Compressing Deep Neural Networks With Learnable Regularization ](https://openreview.net/forum?id=r1e1q3VFvH)

* [Learning Sparsity and Quantization Jointly and Automatically for Neural Network Compression via Constrained Optimization](https://openreview.net/forum?id=SkexAREFDH)

* [Auto Network Compression with Cross-Validation Gradient](https://openreview.net/forum?id=r1eoflSFvS)

---

* [Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning ](https://openreview.net/forum?id=S1ewjhEFwr)

* [MaskConvNet: Training Efficient ConvNets from Scratch via Budget-constrained Filter Pruning](https://openreview.net/forum?id=S1gyl6Vtvr)
	* 和我们还有一些相似的文章

* [Continual Learning via Neural Pruning](https://openreview.net/forum?id=BkeJm6VtPH)

* [Prune or quantize? Strategy for Pareto-optimally low-cost and accurate CNN ](https://openreview.net/forum?id=HkxAS6VFDB)
	* 从pareto最优切入,有点意思

* [The Generalization-Stability Tradeoff in Neural Network Pruning](https://openreview.net/forum?id=B1eCk1StPH)

* [On Iterative Neural Network Pruning, Reinitialization, and the Similarity of Masks ](https://openreview.net/forum?id=B1xgQkrYwS)

* [Meta-Learning with Network Pruning for Overfitting Reduction](https://openreview.net/forum?id=B1gcblSKwB)

* [EMS: End-to-End Model Search for Network Architecture, Pruning and Quantization](https://openreview.net/forum?id=Bkgr2kHKDH)

* [Adversal Network Prunning](https://openreview.net/forum?id=SJe4SJrFDr)

* [Pruning Depthwise Separable Convolutions for Extra Efficiency Gain of Lightweight Models ](https://openreview.net/forum?id=B1g5qyHYPS)

* [Boosting Ticket: Towards Practical Pruning for Adversarial Training with Lottery Ticket Hypothesis](https://openreview.net/forum?id=Sye2c3NYDB)
	* 蹭蹭蹭

* [FNNP: Fast Neural Network Pruning Using Adaptive Batch Normalization](https://openreview.net/forum?id=rJeUPlrYvr)

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191220164603.png)

