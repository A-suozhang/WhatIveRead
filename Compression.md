
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

### Fracs
* Other Methods(Weight Buffer)
	* int8 Training - 32fp BUffer
	* FP8 Training with Stochastic Rounding - 16bit BUFFER
* BN Params Are The Main Problem For Acc Decay in Low-bit Training

