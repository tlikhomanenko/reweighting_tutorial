#Tutorial on BDT reweighter

We are considering reweighting problem (introduction to this problem can be found in the [post](http://arogozhnikov.github.io/2015/10/09/gradient-boosted-reweighter.html)) and show how parameters of classifier/regressors tuning affect on the reweighting rule reconstruction.

Tutorial notebook consists of several parts:
* Neural network parameters: demonstration how these parameters influence reweighting rule;
* Variance and bias errors discussion
* BDT reweighter tuning 
* real use-case in high energy physics, HEP: reweighting problem for sPlot data, when weights (that can be negative) are defined for the target distribution.

In the experiments we use [`hep_ml`](https://github.com/arogozhnikov/hep_ml) library and [carl](https://github.com/diana-hep/carl) library.
