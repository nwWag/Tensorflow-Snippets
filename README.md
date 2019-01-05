# Recurrent Convolutional Hierarchical Attention Network 
## (Sorry for the name.)
This is still ongoing work. The network is an adaption of http://www.aclweb.org/anthology/W18-3002 by Gao et al. Using conv layers makes it quicker and more reliable. However as a next step applying the cam trick (see below) allows to find evidence for activations and allows to give witnesses. In contrast to attention, which is certainly a good approximation for a witness. Nonetheless the most interesting challenge is developing an active learning algorithm for selecting the "right" words out of a text, using the cam witness.

# SVM Layer
This layer implements as an alternative to softmax regression a Support Vector Machine layer.
This is done according to https://arxiv.org/abs/1306.0239 by Tang.

The layer consits of a differentiable quadratic hinge loss as well as a SVM activation.
The layer was pluged into a straightforward dense layer network for MNIST, yielding approximately same results as softmax.

# Class Activation Mapping
This network implements the class activation mapping http://cnnlocalization.csail.mit.edu by Zhou et al..
Trained a few seconds on on Google's Tesla K80. The used dataset is the fashion MNIST. 
Thereby interesting result have been generated even on such an easy dataset (see below, yellow for strong evidence, green middle, blue low).

![](https://i.imgur.com/VTCAh2J.png)

![](https://i.imgur.com/g4Olt1Q.png)

![](https://i.imgur.com/MdMjKWO.png)

![](https://i.imgur.com/QktDomP.png)

# Residual Autoencoder
Each trained approximately 5 min. on Google's Tesla K80.

## Super Resolution
![](https://i.imgur.com/Pj7vHYs.png)

## Autoencoding
![](https://i.imgur.com/1p0P1CO.png)
