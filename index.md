## MultI-chanNel Deep FeatUre Learning for intrusion detection (MINDFUL)

**MINDFUL** is an Deep Learning model that performs a binary classification by combining autoencoders and 1D Convolutional Neural Network. 

![MINDFUL](https://raw.githubusercontent.com/gsndr/MINDFUL_NET/master/MINDFUL.png)

[Read more: Multi-Channel Deep Feature Learning for Intrusion Detection](https://ieeexplore.ieee.org/document/9036935) 



```
class MindfulNET.MINDFUL.MIDFUL_NET(dsConfig, autoencoderA=None,autoencoderN=None,model=None)
```
### Parameters:
* **dsConfig**: a dictionary of parameters ds={'pathModels':'models/CICIDS2017/', 'testName' : 'CICIDS2017'} that indicate the path and the name of saved files and models
* **autoencoderA**: path of learned autoencoder fo class with label 0
* **autoencoderA**: path of learned autoencoder fo class with label 1
* **model**: path of learned 1DCNN


The default path of Deep Learning models is None. If the path of a model is setted the framework load the learned model from path otherwise the framework find the better model by performing a

## Methods
```
fit(X, y)
```
Fit the model according to the given training data

### Parameters:
* **X**: array of shape (n_samples, n_features)
* **Y**:  target array of shape (n_samples,) 

### Returns:
 * **autoencoderA**: learned autoencoder for class with label 0
 * **autoencoderN**: learned autoencoder for class with label 1
 * **autoencoderN**: learned 1DCNN
 


```
predict(X)
```
Predict class for X.

### Parameters:
* **X**: array of shape (n_samples, n_features)

### Returns:
 * **y_pred**: array-like of predicted values shape (n_samples,)

```
predict_proba(X)
```
Predict probabilities for samples in X.

### Parameters:
* **X**: array of shape (n_samples, n_features)

### Returns:
 * **y_prob**: array of the class probabilities of the input samples of  shape (n_samples, n_classes) 


## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Hyperopt](http://hyperopt.github.io/hyperopt/)
* [Hyperas](https://github.com/maxpumperla/hyperas)


## How to use
Here is a simple example of how to make a MINDFUL object:


## Demo Code
As a quick start, a demo script is provided in [example.py](https://github.com/gsndr/MINDFUL_NET/blob/master/example.py). You can either run it directly or enter the following into your python console
``` python3 example.py -d CICIDS2017 -i MINDFUL.conf ```

## Cite
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
