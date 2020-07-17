
# Semi-Supervised Learning

> Personal Digest of Papers in the field Semi-Supervised Learning

> All Work Could Find Code On [PaperWithCode](https://paperswithcode.com/)

> [Related Post](https://a-suozhang.xyz/2020/03/20/Self-Supervised-Learning/)

## Abstract

* 本人较为主观的对自己看过的SemiSupervised的方法做一个简单的梳理，4FutureReference
* 描述的问题是在大量数据Unlabel仅具有少量Label的时候，完成训练(以争取达到和FullySupervised相类似的效果)，主要的实验Seeting有比如
  * 10% / 1% Label of CIFAR-10/SVHN/ImageNet
  * (也有一些文章会尝试更少的Label数目)
  * [Leading Board](https://paperswithcode.com/sota/semi-supervised-image-classification-on-2)
* SemiSupervised问题的分析方式，按照个人理解相对直接的分为以下几种
  1. 基于Regularization的训练Trick的
     * 通过约束训练过程，作用于Classifier的决策边界，对Representation的影响是通过Gradient间接达到的
  2. 基于Augmentation入手的
     * 严格来说和第一类类似，通过设计好的Augmentation方式来获得更好的分类器
  * 以上的可以看作是从Classifier途径优化，而后者相当于是对Representation做了一个优化
  3. 基于Generative的
     * 利用生成模型去完成一个图像重建的任务Reconstruction,用这个任务的效果来衡量Feature的好坏 
  4. Representation Learning/Self-Supervised/Unsupervised Learning相关
     * Unsupervised/Representation Learning的引入，通过引入一个无监督完成SurrogateTask来衡量representation好坏
     * (3-中的Generative方法也可以认为是任务是Reconstruction )

---

## Genre

### 1. 基于Regularization的训练Trick的
  * 最朴素的Pseudo Label - 用Pretrain或者Half-Train的模型给Unlabel的数据打上标签
  * EntMin(认为决策边界不应该穿过高密度区域-PushBackDecisionBoundary，所以Unlabel数据上的熵应该比较小) 
  * [VAT(Consistent Regularization)](https://arxiv.org/abs/1704.03976)
    * [Temporal Ensembling](https://arxiv.org/abs/1610.02242)
    * [MeanTeacher](https://arxiv.org/abs/1703.01780)
  * [MixUp](https://arxiv.org/pdf/1710.09412.pdf) (输入两个Sample的叠加，输出为其对应Label的叠加)


### 2. 基于Augmentation入手的
  * [FixMatch](https://arxiv.org/pdf/1905.02249v2.pdf)/[UDA](https://arxiv.org/abs/1904.12848)等一系列基于Augmentation的叠加的方法(本质都是鼓励图像在一定的扰动下都能比较Consistent) 
  * [enAET](https://arxiv.org/abs/1911.09265) - 用一个VAE去学习出一个Augmentation的组合，输出是加权的Aug

### 3. 基于Generative的
  * 利用生成模型去完成一个图像重建的任务Reconstruction,用这个任务的效果来衡量Feature的好坏 
  * VAE - 看上去好像相对不是那么主流，可能因为VAE最近不是那么Popular
  * GAN - [BigBiGAN](https://arxiv.org/pdf/1907.02544v2.pdf) / [BadGAN](https://arxiv.org/pdf/1705.09783v3.pdf)
  
### 4. Representation Learning/Self-Supervised/Unsupervised Learning相关

  * 随着19年末，Kaiming的Moco提出以及Unsupervised的发展，大家将Unsupervised的方法用到Semi中，成为新SOTA,该领域在[这篇Post](https://a-suozhang.xyz/2020/03/20/Self-Supervised-Learning/)中有一些信息：其主要思想大部分都是通过寻找一个无监督就可以完成的Surrogate Task，用它的完成好坏来衡量Feature(Representation)选取的好坏问题 
  * CPC(Contrastive Predictive Coding):思想是用现在的Feature去训练一个预测未来数据的任务，以获得好的Representation  
  * [Data Efficient Classification with CPC](https://arxiv.org/pdf/1905.09272v2.pdf) 在Classification场景下的一个CPC应用，通过一个预测embedding位置的任务
  * [S4L-SelfSupervisedSemiSupervisedLearning](https://paperswithcode.com/paper/190503670) 将SelfSupervised中的两种方法与Semi结合，主要贡献在于
    * 文章还总结了一些其他方式的SSL的方式，并给了一系列复现
  * [MoCo](http://arxiv.org/pdf/1911.05722.pdf) Kaiming的知名度很高的文章，将Contrastive Learning的问题抽象为了一个Dictionary Learning，并且利用Momentum Update更新的方式提升了性能
  * [SimCLR](https://arxiv.org/pdf/2002.05709v1.pdf) Hinton的新工作





---

## Paper List

1. [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)
  * 2018
  * [Code](https://github.com/brain-research/realistic-ssl-evaluation)
  * Ian Goodfellow
  * Evaluation For SSL Methods  

2. [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)
   * 2016
   * ICLR 2017
   * Nvidia
   * Consistency Training + Temporal Ensemble
   * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120202926.png)
3. [Mean-Teachers Are Better Role-models](https://arxiv.org/abs/1703.01780)
   * Curious AI
   * Change Temporal Ensemble in Pi-Model into Weight-Averaging 
   * [Blog](http://a-suozhang.xyz/2019/10/28/MeanTeacher/)
4. [VAT](https://arxiv.org/abs/1704.03976)
   * Virtual Adversal Learning Regularization Term
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120203446.png)
   * 2-Moon
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120203347.png)

5. [Mixup](https://arxiv.org/pdf/1710.09412.pdf)
   * MIT & FAIR
   * originally proposed to increase robustness against adversal examples
   * Methodology
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120194906.png)
     * Fitting One-Hot - Fitting Linear Combination

6. [ICT](https://arxiv.org/abs/1903.03825)
   * IJCAI 2019
   * Bengio
   * Bring Mixup For Consistency Training
   * **VAT** not suitable for large scale Data
   * Consistecy Training(Modified) encourages the prediction at **an interpolation** of unlabeled points to be consistent with the interpolation of the predictions at those points
   * moves the decision boundary to low-density regions of the data distribution
   * Methodology
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120193417.png)
   * Two-Moons Problem
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120191340.png)
7. [SWA - There Are Many Consistent Explanation, Why Should You Avergage](https://arxiv.org/pdf/1806.05594v3.pdf)
   * [Code](https://github.com/benathi/fastswa-semi-sup)
   * Cornell
   * ICLR 2019
   * SGD Could Not Converge Well For Consistency Loss(Taking  Big Step & Change Prediction), **Changed Optim**
   * *Simplifying Pi-Model*
     * Apply Input Pertubation : The Consistency Loss is The Input-Output Jacobi-Norm
     * Apply Weight Pertubation: The Consistency Loss Is The EigenValue For Hessian Matrix
   * Inspired By Weight-Averaging In Pi-Model Boost Performance(Even Bigger in Supervised Case)
     * Modified SWA(Stochastic Weight Averaging)
       * Cyclic Cosine LR

8. [MixMatch](https://arxiv.org/abs/1905.02249)
   * Google Research / Ian Goodfellow
   * [Code](https://github.com/google-research/mixmatch)
   * K Augmentation - Average - Sharpen
     * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120204429.png)
   * *Thinking About Entropy Minimization*
     * Encourage Model Giving Low-Entropy Label For Unlabeled Data
     * Pseudo Label - Change Soft conf into Hard Label)

9. [UDA(Unsupervised Data Augmentation)](https://arxiv.org/abs/1904.12848)
   * Combine Lots Of Data AUG
   * [Blog](http://a-suozhang.xyz/2019/11/12/UDA/)

10. [EnAET: Self-Trained Ensemble AutoEncoding Transformations](https://arxiv.org/abs/1911.09265)
    * [Code](https://github.com/wang3702/EnAET)
    * Self-Ensmeble + AutoEncodingTransform + MixMatch
    * Amazing Results with extreme few label

* (Some More paper digest are not concluded here, may check [related_posts](https://a-suozhang.xyz/2020/03/20/Self-Supervised-Learning/))

