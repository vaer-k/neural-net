# neural-net
A neural network implemented in Python to classify handwritten numerical digits from MNIST data

To run the network, first clone this repo and ensure that numpy and pandas are installed on your machine. Then simply start an iPython session, import the `nn` module, and call `fit()` on an instantiated network:

```
import nn
clf = nn.NeuralNetwork()
clf.fit()
```

This will automatically use the MNIST dataset and report the results of evaluation on a test using default parameters.
