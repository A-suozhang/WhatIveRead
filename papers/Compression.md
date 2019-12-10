
> Here Contains Some Paper For NN Qantization & Compression

# Quantization

## [Per-Tensor Fixed Point Quantization Of Back-Prop Alogorithm]()

### Abstract (What Have It Done)
* Precision Assignment Methodology for NN
	* Feedforward : Weight & Activation
	* FeedBacl: Gradient(Both For Weight & Activation) & Weight Accumulate
* None Of Works B4 Propose **Realisitc Fixed-Point DNN Training**
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191202164104.png)
* Quantize During Training, NO NEED FOR FLOAT TRAINIG
	* Means Must Predict Bitwidth Only According to Data Statistics

### Fracs (Fracs Knowledge About This FIeld but not directly related to this paper)

> *Some Related Work May Be Added Here Too*

* QFL-Quantize Gradients-For Distributed Training ,but not Saving Computation
* Differnet Variable's Quantize BIT Are Independet and has unquantifiable Trade-Off
* Dynamic Range For Some Quantize Method is not Acquireable
* The "FlexPoint" Data Format: 5 shared exponential bit and per-tensor Dynamic range Tracking

### Evaluation
* A Lot Of Math And Thoughts To Give The **Minimal** BItwidth Needed


## [Hybrid 8 Floating Point Training And Inference For DNN]()

* NIPS 2019

### Abstract
* HPF(Hybrid 8 Floating Point)
* Contribution
	* 8bit Buffer For Weight Update(Outstanding)
	* Try On Other Domains like NLP
* Contains Theoretical Analysis On HPF Type
	* The Mismatch Probablity Sets A Upper Bound For Quantized Training
		* *Explains Why Choosing [1,4,3]*

### Implemention
* Hybrid FP8: 4 Exponential bit + 3 Mantisa BIT (5-2 For BackProp)
* Also Dynamic Weight Range Is Too Big, So Using Weight-Clipping
* When Last FC Layer is big(in NLP), Softmax Could Blur(Biggest Value Quantzied To Same Value)
* 2 Additional Modification
	* BN: 

### Fracs
* Other Methods(Weight Buffer)
	* int8 Training - 32fp BUffer
	* FP8 Training with Stochastic Rounding - 16bit BUFFER
* BN Params Are The Main Problem For Acc Decay in Low-bit Training

## [Training DNN with 8-bit Floating Point Numbers]()

### Abstract 
* 8 Bit Training with 8-Bit Floating Point
	* FP8 (1,5,2) - FP16 (1,6,9)
* Weight Decay Buffer 16 bit (Before this work 32 bit accul)
* Special Technique
	* Chunk-Based Accumul (Without Which Fail to Converge)
		* 8bit Mult & 16 Bit ADD
	* Floating Point Stochastic Rounding
![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191203095523.png)

### Fracs
* Swamping: Change to small for 8bit to represent(in Large-to-small addition)
	* Accul Element are not zero-mean
	* Some Elements are very big (Why Current Device need a buffer larger than 32bit)
* Nearest Rounding discard the LSB infromation
* Last Layer Sensitive to Quantize (Error is amplified as exponential through Softmax
* Input Image are senstive(the color is represented in 8-bit but FP8 Not Enough Mantissa Bit For 8))
* Computing Pattern : GEMM(Matrix Mul) & AXPY(3 Vector Add)-L2reg-Momentum-WeightUpdate

## [Qunatization and Training For Efficient Interger Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)

* CVPR2018

## Abstract 
* Challenging MobileNet Instead Of Redundant VGG & AlexNet
* Contribution
	* Quantize Scheme: 8bit WA
	* Quantized-Inference framework on CPU
	* Quantized Training Framework as Co-Design (**WE NEED**) 

## Implemention
* Constant S(Scale-float) Z(Zero Point) q(quantizaed Value) r(real Value)
	* r = S(q-Z)
* Quantize
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191204160613.png)
* Translate to Integer Multplication
	* Calculating Float Scale in Advance
* Fusing The Layer Together, (Merge Bias Add & Activation into Matmul)
	* **Floating Scale**!
* Training
	* Learning Q-Range(Scale & Zero-Point)
		* Weights [min,max]
		* Using EMA (Also Disable At Start to gain a Stable State)
* Batch-Norm Folding
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191204162634.png)


## Fracs
* Float Training & Finetune Fails For Small Network
	* Outlier Weight Values
	* largeDifference in output channel distribution
* 
