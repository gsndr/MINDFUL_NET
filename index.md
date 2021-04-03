## MultI-chanNel Deep FeatUre Learning for intrusion detection (MINDFUL)

**MINDFUL** is an Deep Learning model that performs a binary classification by combining autoencoders and 1D Convolutional Neural Network. 

![MINDFUL](https://raw.githubusercontent.com/gsndr/MINDFUL_NET/master/MINDFUL.png)

[Read more](https://ieeexplore.ieee.org/document/9036935) 

Please cite our work if you find it useful for your research and work.
```
  @ARTICLE{9036935, 
  author={G. {Andresini} and A. {Appice} and N. D. {Mauro} and C. {Loglisci} and D. {Malerba}}, 
  journal={IEEE Access}, 
  title={Multi-Channel Deep Feature Learning for Intrusion Detection}, 
  year={2020}, 
  volume={8}, 
  number={}, 
  pages={53346-53359},}
```

```
class MindfulNET.MINDFUL.MIDFUL_NET(dsConfig, config, autoencoderA=None,autoencoderN=None,model=None)
```

## Methods
```
fit(X, y)
```
Fit the model according to the given training data

```
predict(X)
```
Predict class labels for samples in X.

```
predict_proba(X)
```
Predict probabilities for samples in X.

