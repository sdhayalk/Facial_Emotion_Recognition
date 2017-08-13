# Emotion Detection from Facial Images using Deep Learning

Implementation of convolutional neural network model similar to VGG-D that can detect emotions given facial images. Achieved an accuracy of 65% on the FER-2013 dataset using GPU for training and testing.

### Prerequisites
Python, TensorFlow, Keras (and other libraries such as NumPy, etc.). If using GPU for training, you will need an NVIDIA GPU card with software packages such as CUDA Toolkit 8.0 and cuDNN v5.1. See [here](https://www.tensorflow.org/install/install_linux) for more details.

Download the FER-2013 dataset from [here](http://www-etud.iro.umontreal.ca/~goodfeli/fer2013.html).

## Input
Input are various 48x48 resolution grayscale images (one channel), along with label corresponding to one of seven emotions such as 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. 

## Results
The accuracy achieved was 65%. I trained and tested on the cloud using FloydHub. Including Batch Normalization and Dropout layers after Convolution-Convolution-Pooling blocks drastically increased the accuracy and overcame the problem of only one class being predicted for all examples. Also omitted one block of Convolution-Convolution-Pooling and fully connected layer.

### References and Acknowledgments
1] Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Zhou, Y. (2013, November). Challenges in representation learning: A report on three machine learning contests. In International Conference on Neural Information Processing (pp. 117-124). Springer, Berlin, Heidelberg.

2] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

3] Pramerdorfer, C., & Kampel, M. (2016). Facial Expression Recognition using Convolutional Neural Networks: State of the Art. arXiv preprint arXiv:1612.02903.
