
> Here Contains Some Paper For NN Qantization & Compression

# Quantization

## [Per-Tensor Fixed Point Quantization Of Back-Prop Alogorithm]()

### Abstract (What Have It Done)
* Precision Assignment Methodology for NN
	* Feedforward : Weight & Activation
	* FeedBacl: Gradient(Both For Weight & Activation) & Weight Accumulate
* None Of Works B4 Propose **Realisitc Fixed-Point DNN Training**
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191202164104.png)

### Fracs (Fracs Knowledge About This FIeld but not directly related to this paper)

> *Some Related Work May Be Added Here Too*

* QFL-Quantize Gradients-For Distributed Training ,but not Saving Computation
* Differnet Variable's Quantize BIT Are Independet and has unquantifiable Trade-Off
* Dynamic Range For Some Quantize Method is not Acquireable
* The "FlexPoint" Data Format: 5 shared exponential bit and per-tensor Dynamic range Tracking

### Evaluation
* A Lot Of Math And Thoughts To Give The **Minimal** BItwidth Needed
