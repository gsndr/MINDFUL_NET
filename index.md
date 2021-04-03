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


## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)


## How to use
Here is a simple example of how to make a MINDFUL object:

## Configuration file
MINDFUL needs as input a path to the configuration file. 
A sample of the configuration file is stored in __MINDFUL.conf__  file 


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase on original date
    LOAD_AUTOENCODER_ADV = 1 #if 1 the autoencoder for attacks items  is loaded from models folder
    LOAD_AUTOENCODER_NORMAL = 1 #if 1 the autoencoder for normal items  is loaded from models folder
    LOAD_CNN = 1  #if 1 the classifier is loaded from models folder
    VALIDATION_SPLIT #the percentage of validation set used to train models
```

## Demo Code
