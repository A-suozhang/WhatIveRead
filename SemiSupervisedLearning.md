
# Semi-Supervised Learning

> All Work Could Find Code On [PaperWithCode](https://paperswithcode.com/)

## [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)

* 2018
* [Code](https://github.com/brain-research/realistic-ssl-evaluation)
* Ian Goodfellow
* Evaluation For SSL Methods  

---

## [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)

* 2016
* ICLR 2017
* Nvidia
* Consistency Training + Temporal Ensemble
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120202926.png)

## [Mean-Teachers Are Better Role-models](https://arxiv.org/abs/1703.01780)

* Curious AI
* Change Temporal Ensemble in Pi-Model into Weight-Averaging 
* [Blog](http://a-suozhang.xyz/2019/10/28/MeanTeacher/)

## [VAT](https://arxiv.org/abs/1704.03976)

* Virtual Adversal Learning Regularization Term
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120203446.png)
* 2-Moon
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120203347.png)

## [Mixup](https://arxiv.org/pdf/1710.09412.pdf)

* MIT & FAIR
* originally proposed to increase robustness against adversal examples
* Methodology
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120194906.png)
  * Fitting One-Hot - Fitting Linear Combination

## [ICT](https://arxiv.org/abs/1903.03825)

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


## [SWA - There Are Many Consistent Explanation, Why Should You Avergage](https://arxiv.org/pdf/1806.05594v3.pdf)

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

## [MixMatch](https://arxiv.org/abs/1905.02249)

* Google Research / Ian Goodfellow
* [Code](https://github.com/google-research/mixmatch)
* K Augmentation - Average - Sharpen
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191120204429.png)
* *Thinking About Entropy Minimization*
  * Encourage Model Giving Low-Entropy Label For Unlabeled Data
  * Pseudo Label - Change Soft conf into Hard Label
* [Blog](http://a-suozhang.xyz/2019/11/12/UDA/)

## [UDA(Unsupervised Data Augmentation)](https://arxiv.org/abs/1904.12848)

* Combine Lots Of Data AUG
* [Blog](http://a-suozhang.xyz/2019/11/12/UDA/)

### [EnAET: Self-Trained Ensemble AutoEncoding Transformations](https://arxiv.org/abs/1911.09265)

* [Code](https://github.com/wang3702/EnAET)
* Self-Ensmeble + AutoEncodingTransform + MixMatch
* Amazing Results with extreme few label

