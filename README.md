# Evolving Convolutional Networks

This project is an experiment in using an evolutionary approach for constructing convolutional networks with [TensorFlow](https://www.tensorflow.org/). 

It was created for the Udacity [Machine Learning Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree-by-google--nd009) Capstone Project.

# Overview

The project a combination of Python 2.7 modules and Jupyter notebooks, using NumPy and TensorFlow. A couple of SciPy image processing modules are also used, specifically scipy.ndimage and scipy.misc for image IO. Scikit-image.color is used for colorspace conversion.

## Python Modules
TensorFlow models are typically built via code that constructs a computation graph of operation nodes and connects them together. This allows TensorFlow to be very general, but it is not an ideal representation for random mutation. Several Python modules were built to define a representation and the tools to randomly mutate it.

In typical convolutional networks, each layer can include operations like dropout and RELU. Some of these operations are only wanted during training. Convolutional and fully connected network layers require parameters that must be initialized. The code to manage these details is in **convnet.py**. Each of the operations and optimizers are wrapped in small objects that set up the parameters in the graph and connect the operations together.

**convevo.py** uses these primitives to construct layers that are made up of multiple operations. For example a fully connected layer will include a matmul operation plus optionally any or all of RELU, add (if biased), dropout (for training only) and L2 loss (for regularization).

These layers can be combined into a LayerStack object, which also specifies which optimizer to use and the learning rate. It can be serialized to XML and supports cross breeding and mutation. It uses **mutate.py** to randomly alter layer properties (eg, patch size, dropout rate, initialization options), duplicate or remove layers, switching optimizers and alter learning rate.

The LayerStack supports zero or more convolution or pooling layers, followed by zero or more ‘deconvolution’/expand layers, then optionally flattening and then zero or more fully connected layers.

It can also automatically adjust certain values to be compatible with different input and output shapes, and to ensure other constraints are satisfied (e.g. stride <= patch size). However it cannot account for issues like resource constraints such as memory size.

Cross breeding combines two layer stacks by randomly taking the first several layers of a given layer type from one stack and splicing that with the last several layers from the other.

**darwin.py** provides a general system for running evolutionary experiments. And finally **improc.py** contains utilities for image processing such as decoding and encoding depth data.

## Jupyter Notebooks

Instructions for download of data for depth reconstruction can be found in depth_setup. Unfortunately the automated download system is both slow and error prone, so the recommended route is to download and decompress the zip files manually.

The notebook **notMNIST_setup** automatically downloads and processes the notMNIST dataset, and provides some visualization of the data.

Constructing and running convnet operations and convevo LayerStacks is demonstrated in the stack test notebook, and a demonstration of the darwin module can be found in the notebook darwin_test.

The actual learning and evolution takes place in **depth_classy** and **depth_pyndent** for depth reconstruction, and in **notMNIST_evo**. Each of these does essentially the same things - construct the TensorFlow graph corresponding to a LayerStack, load and batch the data, run the training data through the graph, calculate metrics, and finally set up an evolutionary process over a population of LayerStacks.

In depth_classy training batches consist of a set of 101 by 101 pixels patches for the inputs with either just the luminance channel or luminance plus chroma, and labels consisting of a one hot encoding of the depth class, along with the normalized depth value. The depth classes use ranges of increasing size, since both the dataset values and sensor accuracy are skewed closer to the camera, and so linear depth ranges would limit precision in the range of real interest. A set of images is loaded into memory and batch is created by taking one patch from each image, restricted to locations where the depth values are valid at the center of the input patch. The loss value for training is the cross entropy between the softmax of the network output and the target label, plus the mean squared difference between the normalized sensor value and the predicted depth, plus any L2 loss contributions based on the LayerStack settings. A tuning factor is used to adjust the relative weight of the two main loss contributions.

In addition, the depth_classy notebook sets up a second mechanism for running the Tensorflow graph which systematically reconstructs the complete depth image corresponding to an input image. Since depth_classy only predicts the center pixel for an image patch, this leaves a ‘frame’ of the original depth data around the edges. This code produces three different versions of the reconstructed depth. One is the raw depth prediction. The second takes the softmax label prediction and interpolates the depth by taking the sum of the softmax weight for each for each class multiplied by the midpoint of the range. The third image uses the midpoint of the class range with the largest predicted softmax weight. These reconstructions are used to calculate accuracy and error metrics for the test set.

For depth_pyndent, the input is the entire CIELAB image, or optionally just the luminance, and the target output is the entire depth result. To deal with missing depth data the predicted values from the model output are copied over the NaN values in the target before calculating the mean squared difference. This results in zero loss contribution from pixels with missing depth. Batch size is limited (typically just a single image) due to memory constraints.

Finally notMNIST_evo simply loads and batches the preprocessed data generated from notMNIST_setup. It also includes some testing debugging aids used during development of the evolutionary system.
