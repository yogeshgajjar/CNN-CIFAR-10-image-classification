<!-- # CIFAR-10-image-classification  -->

# CNN based CIFAR-10 Image Classifier 

This repository contains two different CNN image classifier trained using two different architectures. The first model is trained on All-CNN architecture which achieves 90% accuracy using Keras framework. The second model is trained on LeNet-5 architecture which achieves 74% accuracy using PyTorch.  

## All - CNN YGNet architecture using Keras 

This model is motivated from the [Striving for Simplicity - All Convolution Net](https://arxiv.org/abs/1412.6806) paper. The paper achieves 95.6% accuracy using the All-CNN architecture. My model (YGNet) has few changes in the architecture than the All-CNN architecture used in the paper. I've used max-pooling instead of making it a fully convolutional network. The total number of the trainable parameters remains the same i.e. ~1.3M. The changes used in my architecture were made on purpose to understand the effect and importance of certain layers in the convolutional neural network. 

![YGNet](mycnn_architecture.png "YGNet") 