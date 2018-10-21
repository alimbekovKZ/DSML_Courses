EE-559 – Deep Learning (Spring 2018)
====================================

 [![Idiap's logo](www-pics/logo_idiap.png)](http://www.idiap.ch/) [![EPFL's logo](www-pics/logo_epfl.png)](http://www.epfl.ch/) 

You can find here info and materials for the [EPFL](http://www.epfl.ch) course [EE-559 “Deep Learning”](http://edu.epfl.ch/coursebook/en/deep-learning-EE-559), taught by [François Fleuret](http://www.idiap.ch/~fleuret/). This course is an introduction to deep learning tools and theories, with examples and exercises in the [PyTorch](http://pytorch.org) framework.

These materials were designed for PyTorch versions 0.3.x, which differs slightly from the current stable versions 0.4.x. You can get [updated slides,](https://fleuret.org/ee559/) which are also slightly restructured.

Most of the content remains relevant though, in particular the voice-overs. The main changes are in the parts about tensors, autograd and GPU: new 1.4, 1.5, and 1.6 replace old 1b, new 4.2 replaces the autograd part of old 4, and 6.6 replaces the end of old 6 on GPUs.

[Materials](#materials) | [Information](#information) | [Virtual Machine](#vm) | [Mini-projects](#mini-projects) | [License](#license)

Thanks to Adam Paszke, Alexandre Nanchen, Xavier Glorot, Andreas Steiner, Matus Telgarsky, Diederik Kingma, Nikolaos Pappas, Soumith Chintala, and Shaojie Bai for their answers or comments.

Materials
=========

You will find here slides, handouts with two slides per pages, voice-over videos, and practicals.

A [Virtual Machine,](#vm) and a helper [python prologue](#prologue) for the practical sessions are available below.

Lectures
--------

1.  [Introduction and tensors](#course-1)
2.  [Machine learning fundamentals](#course-2)
3.  [Multi-layer perceptrons](#course-3)
4.  [Convolutional networks and autograd](#course-4)
5.  [Optimization](#course-5)
6.  [Going deeper](#course-6)
7.  [Computer vision](#course-7)
8.  [Under the hood](#course-8)
9.  [Autoencoders and generative models](#course-9)
10.  [Generative Adversarial Networks](#course-10)
11.  [Recurrent networks and Natural Language Processing](#course-11)

### Guest lectures

12.  [Deep Learning in the Real World](#course-12) (Soumith Chintala, Facebook)
13.  [From PyTorch To TensorFlow](#course-13) (Andreas Steiner, Google)
14.  [QuickDraw End2end using TensorFlow](#course-14) (Andreas Steiner, Google)

Lecture 1 (Feb 21, 2018) – Introduction and tensors
---------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-1.png)](dlc-slides-1a-introduction.pdf)

What is deep learning, some history, what are the current applications. torch.Tensor, linear regression.

*   [slides](dlc-slides-1a-introduction.pdf) / [handout](dlc-handout-1a-introduction.pdf) / [video (1h04min)](../videos/dlc-video-1a-introduction.mp4) (part a)
*   [slides](dlc-slides-1b-tensors.pdf) / [handout](dlc-handout-1b-tensors.pdf) / [video (37min)](../videos/dlc-video-1b-tensors.mp4) (part b)
*   [practical](dlc-practical-1.pdf) / [dlc\_practical\_1\_solution.py](dlc_practical_1_solution.py)

Lecture 2 (Feb 28, 2018) – Machine learning fundamentals
--------------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-2.png)](dlc-slides-2-ml-basics.pdf)

Empirical risk minimization, capacity, bias-variance dilemma, polynomial regression, k-means and PCA.

*   [slides](dlc-slides-2-ml-basics.pdf) / [handout](dlc-handout-2-ml-basics.pdf) / [video (1h28min)](../videos/dlc-video-2-ml-basics.mp4)
*   [practical](dlc-practical-2.pdf) / [dlc\_practical\_2\_solution.py](dlc_practical_2_solution.py)

Lecture 3 (Mar 07, 2018) – Multi-layer perceptrons
--------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-3.png)](dlc-slides-3a-linear.pdf)

Linear classifiers, perceptron, linear separability and feature extraction, Multi-Layer Perceptron, gradient descent, back-propagation.

*   [slides](dlc-slides-3a-linear.pdf) / [handout](dlc-handout-3a-linear.pdf) / [video (51min)](../videos/dlc-video-3a-linear.mp4) (part a)
*   [slides](dlc-slides-3b-mlp.pdf) / [handout](dlc-handout-3b-mlp.pdf) / [video (41min)](../videos/dlc-video-3b-mlp.mp4) (part b)
*   [practical](dlc-practical-3.pdf) / [dlc\_practical\_3\_solution.py](dlc_practical_3_solution.py)

Lecture 4 (Mar 14, 2018) – Convolutional networks and autograd
--------------------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-4.png)](dlc-slides-4a-dag-autograd-conv.pdf)

Generalized acyclic graph networks, torch.autograd, batch processing, convolutional layers and pooling, torch.nn.Module.

*   [slides](dlc-slides-4a-dag-autograd-conv.pdf) / [handout](dlc-handout-4a-dag-autograd-conv.pdf) / [video (51min)](../videos/dlc-video-4a-dag-autograd-conv.mp4) (part a)
*   [slides](dlc-slides-4b-modules-batch.pdf) / [handout](dlc-handout-4b-modules-batch.pdf) / [video (46min)](../videos/dlc-video-4b-modules-batch.mp4) (part b)
*   [practical](dlc-practical-4.pdf) / [dlc\_practical\_4\_solution.py](dlc_practical_4_solution.py)
*   [dlc\_practical\_4\_embryo.py](dlc_practical_4_embryo.py)

Lecture 5 (Mar 21, 2018) – Optimization
---------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-5.png)](dlc-slides-5-init-optim.pdf)

Cross-entropy, L1 and L2 penalty. Vanishing gradient, weight initialization, Xavier's rule, loss monitoring. torch.autograd.Function.

*   [slides](dlc-slides-5-init-optim.pdf) / [handout](dlc-handout-5-init-optim.pdf) / [video (1h43min)](../videos/dlc-video-5-init-optim.mp4)
*   [practical](dlc-practical-5.pdf) / [dlc\_practical\_5\_solution.py](dlc_practical_5_solution.py)

Lecture 6 (Mar 28, 2018) – Going deeper
---------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-6.png)](dlc-slides-6-going-deeper.pdf)

Theoretical advantages of depth, rectifiers, drop-out, batch normalization, residual networks, advanced weight initialization. GPUs and torch.cuda.

*   [slides](dlc-slides-6-going-deeper.pdf) / [handout](dlc-handout-6-going-deeper.pdf) / [video (1h34min)](../videos/dlc-video-6-going-deeper.mp4)
*   [mini-projects](#mini-projects)

No lecture (Apr 4, 2018) – Easter break
---------------------------------------

Lecture 7 (Apr 11, 2018) – Computer vision
------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-7.png)](dlc-slides-7-computer-vision.pdf)

Deep networks for image classification (AlexNet, VGGNet), object detection (YOLO), and semantic segmentation (FCN). Data-loaders, neuro-surgery, and fine-tuning.

*   [slides](dlc-slides-7-computer-vision.pdf) / [handout](dlc-handout-7-computer-vision.pdf) / [video (1h43min)](../videos/dlc-video-7-computer-vision.mp4)
*   [mini-projects](#mini-projects)

Lecture 8 (Apr 18, 2018) – Under the hood
-----------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-8.png)](dlc-slides-8-under-the-hood.pdf)

Visualizing filters and activations, smoothgrad, deconvolution, guided back-propagation. Optimizing samples from scratch, adversarial examples. Dilated convolutions.

*   [slides](dlc-slides-8-under-the-hood.pdf) / [handout](dlc-handout-8-under-the-hood.pdf) / [video (1h23min)](../videos/dlc-video-8-under-the-hood.mp4)
*   [mini-projects](#mini-projects)

Lecture 9 (Apr 25, 2018) – Autoencoders and generative models
-------------------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-9.png)](dlc-slides-9-autoencoders.pdf)

Transposed convolution layers, autoencoders, variational autoencoders, non volume-preserving networks.

*   [slides](dlc-slides-9-autoencoders.pdf) / [handout](dlc-handout-9-autoencoders.pdf) / [video (1h27min)](../videos/dlc-video-9-autoencoders.mp4)
*   [mini-projects](#mini-projects)

Lecture 10 (May 2, 2018) – Generative Adversarial Networks
----------------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-10.png)](dlc-slides-10-gans.pdf)

GAN, Wasserstein GAN, Deep Convolutional GAN, Image-to-Image translations, model persistence.

*   [slides](dlc-slides-10-gans.pdf) / [handout](dlc-handout-10-gans.pdf) / [video (1h28min)](../videos/dlc-video-10-gans.mp4)
*   [mini-projects](#mini-projects)

Lecture 11 (May 9, 2018) – Recurrent networks and Natural Language Processing
-----------------------------------------------------------------------------

[![Thumbnail made from a slide](www-pics/thumb-11.png)](dlc-slides-11-rnn-nlp.pdf)

Back-propagation through time, gating, LSTM, GRU. Word embeddings, sentence-to-sentence translation.

*   [slides](dlc-slides-11-rnn-nlp.pdf) / [handout](dlc-handout-11-rnn-nlp.pdf) / [video (1h17min)](../videos/dlc-video-11-rnn-nlp.mp4)
*   [mini-projects](#mini-projects)

Lecture 12 (May 16, 2018) – Deep Learning in the Real World
-----------------------------------------------------------

**Guest speaker:** [Soumith Chintala](https://research.fb.com/people/chintala-soumith/) (Facebook)

![Speaker in front of the audience](www-pics/thumb-12.png)

Large data-sets and models, effectively parallelizing on GPUs, pythonless deployment. torch.distributed, ONNX, exporting to Caffe2.

*   [slides](dlc-slides-12-soumith-chintala.pdf) / [video (1h26min)](../videos/dlc-video-12-soumith-chintala.mp4)
*   [mini-projects](#mini-projects)

Lecture 13 (May 23, 2018) – From PyTorch To TensorFlow
------------------------------------------------------

**Guest speaker:** Andreas Steiner (Google)

![Speaker in front of the audience](www-pics/thumb-13.png)

ML research at Google, mobile ML application example, Low-level TensorFlow Python API (Graph, Session, shapes, variables).

*   [Google Colab for the lecture](https://goo.gl/EKVKWz)
*   [Practical session info sheet](https://goo.gl/rmfoka)
*   [Google Colab for the practical session](https://goo.gl/c9F4oB)

**Restricted to enrolled students:**

*   [handout](https://moodle.epfl.ch/mod/resource/view.php?id=986772)
*   [video (12min)](https://moodle.epfl.ch/mod/url/view.php?id=988074) (part 1)
*   [video (5min)](https://moodle.epfl.ch/mod/url/view.php?id=988075) (part 2)
*   [video (9min)](https://moodle.epfl.ch/mod/url/view.php?id=988076) (part 3)
*   [video (31min)](https://moodle.epfl.ch/mod/url/view.php?id=988077) (part 4)
*   [video (8min)](https://moodle.epfl.ch/mod/url/view.php?id=988078) (part 5)

Lecture 14 (May 30, 2018) – QuickDraw End2end using TF
------------------------------------------------------

**Guest speaker:** Andreas Steiner (Google)

Transforming QuickDraw data, estimator and experiment interfaces, CNN/RNN classifiers, CloudML.

*   [Practical session info sheet](https://goo.gl/KJTXLT)

**Restricted to enrolled students:**

*   [handout](https://moodle.epfl.ch/mod/resource/view.php?id=987453) (restricted to EPFL)
*   [video (11min)](https://moodle.epfl.ch/mod/url/view.php?id=988687) (part 1)
*   [video (10min)](https://moodle.epfl.ch/mod/url/view.php?id=988688) (part 2)
*   [video (11min)](https://moodle.epfl.ch/mod/url/view.php?id=988689) (part 3)
*   [video (7min)](https://moodle.epfl.ch/mod/url/view.php?id=988690) (part 4)

Information
===========

Pre-requisites
--------------

*   Linear algebra (vector and Euclidean spaces),
*   differential calculus (Jacobian, Hessian, chain rule),
*   Python,
*   basics in probabilities and statistics (discrete and continuous distributions, law of large numbers, conditional probabilities, Bayes, PCA),
*   basics in optimization (notion of minima, gradient descent),
*   basics in algorithmic (computational costs),
*   basics in signal processing (Fourier transform, wavelets).

Documentation
-------------

You may have to look at the python 3, jupyter, and PyTorch documentations at

*   [https://docs.python.org/3/](https://docs.python.org/3/)
*   [http://jupyter.org/](http://jupyter.org/)
*   [http://pytorch.org/docs/](http://pytorch.org/docs/)

Grading
-------

The final grade will be 25% for each of the two [mini-projects](#mini-projects) grade, and 50% for the written exam during the exam session.

Practical session prologue
--------------------------

Helper python prologue for the practical sessions: [dlc\_practical\_prologue.py](dlc_practical_prologue.py)

### Argument parsing

This prologue parses command-line arguments as follows

usage: dummy.py \[-h\] \[--full\] \[--tiny\] \[--force\_cpu\] \[--seed SEED\]
                \[--cifar\] \[--data\_dir DATA\_DIR\]

DLC prologue file for practical sessions.

optional arguments:
  -h, --help           show this help message and exit
  --full               Use the full set, can take ages (default
                       False)
  --tiny               Use a very small set for quick checks
                       (default False)
  --force\_cpu          Keep tensors on the CPU, even if cuda is
                       available (default False)
  --seed SEED          Random seed (default 0, < 0 is no seeding)
  --cifar              Use the CIFAR data-set and not MNIST
                       (default False)
  --data\_dir DATA\_DIR  Where are the PyTorch data located (default
                       $PYTORCH\_DATA\_DIR or './data')

It sets the default Tensor to torch.cuda.FloatTensor if cuda is available (and \--force\_cpu is not set).

### Loading data

The prologue provides the function

load\_data(cifar = None, one\_hot\_labels = False, normalize = False, flatten = True)

which downloads the data when required, reshapes the images to 1d vectors if flatten is True, narrows to a small subset of samples if \--full is not selected, moves the Tensors to the GPU if cuda is available (and \--force\_cpu is not selected).

It returns a tuple of four tensors: train\_data, train\_target, test\_data, and test\_target.

If cifar is True, the data-base used is CIFAR10, if it is False, MNIST is used, if it is None, the argument \--cifar is taken into account.

If one\_hot\_labels is True, the targets are converted to 2d torch.Tensor with as many columns as there are classes, and -1 everywhere except the coefficients \[n, y\_n\], equal to 1.

If normalize is True, the data tensors are normalized according to the mean and variance of the training one.

If flatten is True, the data tensors are flattened into 2d tensors of dimension N × D, discarding the image structure of the samples. Otherwise they are 4d tensors of dimension N × C × H × W.

### Minimal example

import dlc\_practical\_prologue as prologue

train\_input, train\_target, test\_input, test\_target = prologue.load\_data()

print('train\_input', train\_input.size(), 'train\_target', train\_target.size())
print('test\_input', test\_input.size(), 'test\_target', test\_target.size())

prints

data\_dir ./data
\* Using MNIST
\*\* Reduce the data-set (use --full for the full thing)
\*\* Use 1000 train and 1000 test samples
train\_input torch.Size(\[1000, 784\]) train\_target torch.Size(\[1000\])
test\_input torch.Size(\[1000, 784\]) test\_target torch.Size(\[1000\])

Virtual machine
===============

A Virtual Machine (VM) is a software that simulates a complete computer. The one we provide here includes a Linux operating system and all the tools needed to use PyTorch from a web browser (firefox or chrome).

Installation at the EPFL
------------------------

It is already installed on the machines in room CM1 103 for the exercise sessions. You can start it with Windows Start menu → All programs → VirtualBox → VB\_DEEP\_LEARNING.

Installation of the VM on your own computer
-------------------------------------------

If you want to use your own machine, first download and install: [Oracle's VirtualBox](https://www.virtualbox.org/wiki/Downloads) then download the file: [Virtual machine OVA package (large file ~3.6Gb)](https://documents.epfl.ch/users/f/fl/fleuret/www/ova/Deep%20Learning%20VM%200.2.ova) and open it in VirtualBox with File → Import Appliance.

You should now see an entry in the list of VMs. The first time it starts, it provides a menu to choose the keyboard layout you want to use (you can force the configuration later by passing forcekbd to the kernel through GRUB).

**If the VM does not start and VirtualBox complains that the VT-x is not enabled, you have to activate the virtualization capabilities of your Intel CPU in the BIOS of your computer.**

Using the VM
------------

The VM automatically starts a [JupyterLab](http://jupyter.org/) on port 8888 and exports that port to the host. This means that you can access this JupyterLab with a web browser on the machine running VirtualBox at: [http://localhost:8888/](http://localhost:8888/) and use python notebooks, view files, start terminals, and edit source files. Typing !bye in a notebook or bye in a terminal will shutdown the VM.

You can run a terminal and a text editor from inside the Jupyter notebook for exercises that require more than the notebook itself. Source files can be executed by running in a terminal the python command with the source file name as argument. Both can be done from the main Jupyter window with:

*   New → Text File to create the source code, or selecting the file and clicking Edit to edit an existing one.
*   New → Terminal to start a shell from which you can run python.

**Files saved in the VM are erased when the VM is re-installed, which happens for each session on the EPFL machines. So you should download files you want to keep from the jupyter notebook to your account and re-upload them later when you need them.**

This VM also exports an ssh port to the port 2022 on the host, which allows to log in with standard ssh clients on Linux and OSX, and with applications such as [PuTTY](https://www.putty.org/) on Windows. The default login is 'dave' and password 'dummy', same password for the root.

Remarks
-------

Note that performance for computation will not be as good as if you [install PyTorch](http://pytorch.org/) natively on your machine (which is possible only on Linux and OSX for versions < 0.4.0). In particular, the VM does not take advantage of a GPU if you have one.

**Finally, please also note that this VM is configured in a convenient but highly non-secured manner, with easy to guess passwords, including for the root, and network-accessible non-protected Jupyter notebooks.**

This VM is built on a [Linux](https://www.linuxfoundation.org/) [Debian 9.3 “stretch”,](https://www.debian.org/) with [miniconda,](https://conda.io/miniconda.html) [PyTorch 0.3.1,](http://pytorch.org/) [TensorFlow 1.4.1,](https://www.tensorflow.org/) [MNIST,](http://yann.lecun.com/exdb/mnist/) [CIFAR10,](https://www.cs.toronto.edu/~kriz/cifar.html) and many Python utility packages installed.

Mini-projects
=============

Here are the two mandatory mini-projects:

*   Mini-project 1: [Prediction of finger movements from EEG recordings.](dlc-miniproject-1.pdf)
*   Mini-project 2: [Implementing from scratch a mini deep-learning framework.](dlc-miniproject-2.pdf)

License of use
==============

The pdf files and videos on this page are licensed under the [Creative Commons BY-NC-SA 4.0 International License.](by-nc-sa-4.0.txt)

More simply: I am okay with this material being used for regular academic teaching, but definitely not for a book / youtube loaded with ads / whatever monetization model I am not aware of.

[Valid XHTML 1.0 Strict](https://validator.w3.org/check?uri=https%3A%2F%2Ffleuret.org%2Fdlc%2F) · [Valid CSS](http://jigsaw.w3.org/css-validator/validator?uri=https%3A%2F%2Ffleuret.org%2Fdlc%2F) · [No dead link](https://validator.w3.org/checklink?uri=https%3A%2F%2Ffleuret.org%2Fdlc%2F&summary=on&hide_redirects=on&hide_type=all&depth=&check=Check)