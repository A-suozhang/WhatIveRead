# 定义
* Life Long & Online Learning & Incremental  核心在于**可继续的学习**后者加上了在线更新以及内存受限等限制
* Transfer Learning & Multitasking & Few-shot Learning  主要处理的是数据差异大的问题
* Continual DNN时代IL的阐发  更加广义地解释IL

* What's Different In **Incremental Learning**?
    * Dynamic
    * Use Data Streaming (Has Internal Temporal Order) - Could Be Used
    * Adaptive Model Complexity - 例如SVM中的SV数目，NN中的 Hidden Unit数目运行中改变
    * 其训练本质和Stochastic的训练方式类似（SGD,especially Batch），区别在于Hyper-param是依据整个数据分布设定的，而这里数据分布未知且变化（Concept Drift）

# 应用场景  
* **使模型更加贴合用户的使用习惯**

* 终端     
    * No Connection -> Protect Privact 
    * Robotic（Auto-Driving） / IOT设备
* 云端（Big Data） (上线之后的数据和训练数据差距大们需要Adaptive)
    * 推荐系统  
    * 数据降维表征处理 Feature Extraction：PCA，Clustering

# 主要问题（研究方向） *Ranked By Importance*
（由于具体的处理方法因对应算法常有改变，本部分的Solution以解决问题的思想为主）
## Concept Drift
* 最主流解决方案： Ensemble
* Incremental Learning 中的主要问题之一（有时也别描述为 Unstable-Data ）主要可以分为两类： Virtual & Real
    * Virtual: 输入数据的Distribution发生改变
        * 与 Class Imbalance 相关   
            * *采用 Importance Weighting 的思想，对新输入的数据做特殊处理*
        * 当新 Class 会引入之后，输入数据会突变 —— Novelty（奇异）
            * *引入 Novelty Check模块，针对性解决问题*
    * Real: The P(Y|X) Changed
        * Statistically Detect CD(偏硬核统计方法) - Hoelffding Distance(不太有后续)
        * Ensemble Model(集成学习) 融合不同的分类器 （感觉是靠Robustness硬扛）  
        **目前最有效的处理Concept Drift的方法** 

## Catastrophic Forgetting
* 最主流解决方案： 加强学习规则 （Enforce Learning Rules）- Explicit Meta Strategy
* 由于模型的计算量不能无限增大，所以Incremental Learning在接受新知识的同时，也有一个忘记旧知识的过程。“学的越快，忘的也越快”，这里存在一个权衡。（有时该问题也会被叫做 Stability-Plasticly Dilema）
* 目前深度学习中的Catastrophic Forgetting貌似含义和IL中的不太一样，可能有时候更为广义,训练数据和上线之后的实际数的差异大的问题也属于此类,我理解是NN泛化能力不足,Incremental是解决它的方案之一.
    * 在NN中的主流解决方案:    **在Loss中加项去"蒸馏"出老任务的信息并加以保留**
* 在如NN等Connetionist类的模型中比较严重
    * 在新时代处理方式比较多样，在NN领域 有一篇[综述文章](https://openreview.net/forum?id=BkloRs0qK7)介绍

## Memory Bottleneck —— Efficient Memory Model
* 由于存储有限，所以需要尽量的压缩输入数据 
* 主要分为两类 数据表征Explicit/Implicit
    * Implicit （Example/Prototype Based Methods： Combine Human Coginitive Categories & Numerical Features ）
    * Explicit (经常采用短期记忆的方法，只保留一段时间)

## Meta-Parameter Tunning
* 由于输入数据Unstable,导致超参数是变量
    * 让模型更加Robust
    * 采取Meta-Heuristic的方法去调整超参数（比较困难）

# Benchmark
由于该领域较为宽广，处理的问题与使用算法的跨度都比较大，导致没有一个公认的指标（类似Image Classification 的ILVSR）
同时，衡量IL的算法的指标维度较多，有一篇[文章](https://www.sciencedirect.com/science/article/pii/S0925231217315928)对不同算法进行了比较好的对比(对比了截止2018年的各种SOTA的对比，但是不太有NN相关的)

* 主要Evaluate的维度有
    * Accuracy
    * Converge Time
    * Complexity
    * Hyper-Params Optimization

# New Popular Methods
1. LWTA(Local Winner Takes It All) - As An Activation Function
2. EWC(Elastic Weight Consolidation) - As Regularization In Loss Function
3. ICaRL(Incremental Classification and Representation Learning) - A Structure Of Representation Learning Using CNN
4. LwF(Learning Without Forgetting)


# 推荐文章
## 1. Incremental On-line Learning: A Review and Comparison of State of the Art Algorithms  （A Survey On Pre-NN SOTA Methods For IL）
* [Link](https://www.sciencedirect.com/science/article/pii/S0925231217315928)
* Cite：33
* Rank :   :star::star::star::star:
* INTRO:   对前NN时代的IL的SOTA方法进行了多维的可靠对比,同时梳理了他们用到的方法
* Digest: 

## 2. Incremental learning algorithms and applications (Old Survey Of The Field Of IL 2015)
* [Link](https://hal.archives-ouvertes.fr/hal-01418129/)
* Cite：62
* Rank : :star::star::star:
* INTRO: 对整个领域的概况做了详细的梳理，也列举了很多参考文献；缺陷在于年代过于久远
* Digest： 

## 3. Interactive Online Learning for Obstacle Classification on a Mobile Robot 
* [Link](https://ieeexplore.ieee.org/abstract/document/7280610)
* Cite: 17
* Rank: :star::star::star:
* INTRO: 算是一篇比较完整的文章，多方考虑了各种问题（比如Memory Bound），最后有自然场景的应用落地，核心问题是图像的分类问题，采用了I-LVQ
* Digest

***
(以下的文章是与NN有关的IL相关文章,方法与以上的方法差距相对较大,但是思想类似)

## 4. An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks 
* [link](https://arxiv.org/abs/1312.6211)
* Cite: 200+
* Rank: :star::star:
* INTRO: Bengio实验室出的一篇分析论文,主要指出了Dropout训练对于缓解CF的作用,领域开山,意义不是特别大,但是采用的分割数据集方法比较经典(利用Premute分割Mnist1为几个SubTask)
* [Code](https://github.com/goodfeli/forgetting)

## :star: 5. Learning Without Forgetting (ECCV 2016)
* [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8107520)
* Rank: :star::star::star::star:
* INTRO
    * 他们居然自己更新自己的工作还起了一样的名字? [Old Link](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)
    * 2016年那一版是领域的开山之作,比较经典,应用面很广,不仅在IL领域,也涉及TF
    * 用了挺多比较炫技的东西,思想基于Distillation **思想重要,方法其次**
    * 文章内容稍偏杂,分析对比也不止有Incremental Learning

## :star: 6.  iCaRL_Incremental_Classifier(CVPR_2017)
* [link](http://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)
* Cite: 172
* Rank: :star::star::star::star::star:
* INTRO: NN时期的IL的一篇起步的标准文章.采用CNN做Feature Extractor,每一类别给出一个Prototype-Vector以完成分类,给出了一个完整的Workflow,对比如Memory Bound等问题都有所考虑到,之后的文章多有引用
* [Code](https://github.com/srebuffi/iCaRL)   Tensorflow
* [Code](https://github.com/donlee90/icarl)   PyTorch  (这个代码里面有把cifar变icifar的实现)

## :star: 7. EWC - Overcoming CF In NN (NIPS2017)
* [link](https://www.pnas.org/content/114/13/3521.short)
* Cite: 541
* Rank::star::star::star::star::star:
* INTRO: Deepmind出品,提出了EWC,本质是修改Loss函数加上了约束,让网络在训练新任务的时候尽量保持对原任务重要的参数(有一个弹性机制),文章中对分类和RL任务都做了分析.可实现性强,开山文章,经典.
* [Code](https://github.com/stokesj/EWC)

## 8. A COMPREHENSIVE, APPLICATION-ORIENTED STUDY OF CATASTROPHIC FORGETTING IN DNNS (ICLR 2019 )
* [link]()
* Cite: 1 (New!)
* Rank: :star::star::star::star:
* INTRO： 后NN时代关于IL的一篇Survey，设计了一些Sequential Learning的场景（在Visual Classification任务上）介绍了目前NN领域IL的一些方法
    * 介绍了NN时代IL领域的几个著名方法:
        * EWC
        * LWTA
        * IMM [Paper](https://arxiv.org/abs/1312.6211)   这篇文章中说实现意义不大



## 其他文章
* Incremental learning of object detectors without catastrophic forgetting (ICCV 2017; 50 Cite)
    * [link](http://openaccess.thecvf.com/content_iccv_2017/html/Shmelkov_Incremental_Learning_of_ICCV_2017_paper.html) 
    * Object Detcetion的一篇文章
* End2End Incremental Learning(ECCV 2018; Cite 18)
    * [link](http://openaccess.thecvf.com/content_ECCV_2018/html/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.html)
    * 在ICaRL基础上完成了End2End
* PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning(CVPR2018; 43 Cite)
    * [link](http://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html)
    * 与DeepMind的EWC思路类似,利用Over-Param,通过迭代剪枝实现
* Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights(ECCV 2018;Cite 16)
    * [link](https://arxiv.org/abs/1801.06519)
    * 与上面一篇思路类似
* Reinforced Continual Learning(NIPS2018; 12Cite)
    * [link](http://papers.nips.cc/paper/7369-reinforced-continual-learning)
    * Rl的一篇文章
* Large Scale Incremental Learning (ICCV 2019)
    * [link](https://arxiv.org/pdf/1905.13260.pdf)
    * 比较新的一篇,比较大规模的做IL的


# 资源
1. 他人总结的 Incremental Learning Reading List  [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
2. 比IL更广泛的Continual Learning Benchmark  [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark)
3. 数据集 [CoRe50](https://vlomonaco.github.io/core50/benchmarks.html)
    * 很多论文中对数据集的处理方案还是用的Goodfellow文中创造SubTask的方案

# （个人思考）
* 整个领域的发展: 
    * 首先以2013年Bengio和Goodfellow那篇文章的研究开始,人们开始研究NN的CF(其实解决了就能Incremental了) 当时已经有了比如LWTA之类的一些尝试,但是不成熟
    * EWC的提出展示了Incremental的比较好的解决方案,之后的很多算法都基于它
    * ICaRL则是一个比较有代表性的WorkFlow
    * 后面则是各种开花,大家提出各式各样的架构让各种网络Incremental(有些就很玄学了,基于Brain的啊,各种distillatoin的啊,还有Attention机制啊等一些比较花哨的Trick,但是主干基本上还是ICARL的Method)
* Incremental问题的思考
    * Incremetal常与CF相联系,但是CF并不只通过IL解决,同样也可以通过Transfer等方式解决.
    * 很多方法的设计依赖网络的Over-Param,与模型压缩矛盾
* 数据的标注问题，训练需要有Label的数据
    * 在推荐系统等以Human FeedBack作为Label （在自动驾驶中其实也可以把人的行为作为FeedBack，不过没有那么直接）
    * FeedBack的Latency,数据进入之后需要完成动作才能有FeedBack。（其实对于训练来说个人感觉没有那么大所谓，就是对数据的储存提出了要求）
* 数据集的问题:
    * Ian GoodFellow的方法,对于一个数据集分出SubTask(利用Permutation)
    * iCifar 对原本数据集的修改
* Memory Bound的问题:
    * 目前算法只要是Memory有Bound的都拿出来吹自己可实现,还没有落实到具体存储问题

