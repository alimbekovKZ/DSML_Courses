
[Source](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/ "Permalink to Deep Learning Course (Spring 2018)")

# Deep Learning Course (Spring 2018)

You can find here info and materials for the [EPFL][5] course [EE-559 "Deep Learning"][6], taught by [François Fleuret][7]. This course is an introduction to deep learning tools and theories, with examples and exercises in the [PyTorch][8] framework.

These materials were designed for PyTorch versions 0.3.x, which differs slightly from the current stable versions 0.4.x. You can get [updated slides,][9] which are also slightly restructured. 

Most of the content remains relevant though, in particular the voice-overs. The main changes are in the parts about tensors, autograd and GPU: new 1.4, 1.5, and 1.6 replace old 1b, new 4.2 replaces the autograd part of old 4, and 6.6 replaces the end of old 6 on GPUs. 

Thanks to Adam Paszke, Alexandre Nanchen, Xavier Glorot, Andreas Steiner, Matus Telgarsky, Diederik Kingma, Nikolaos Pappas, Soumith Chintala, and Shaojie Bai for their answers or comments.

You will find here slides, handouts with two slides per pages, voice-over videos, and practicals.

A [Virtual Machine,][10] and a helper [python prologue][11] for the practical sessions are available below.

## Lectures

1. [Introduction and tensors][12]
2. [Machine learning fundamentals][13]
3. [Multi-layer perceptrons][14]
4. [Convolutional networks and autograd][15]
5. [Optimization][16]
6. [Going deeper][17]
7. [Computer vision][18]
8. [Under the hood][19]
9. [Autoencoders and generative models][20]
10. [Generative Adversarial Networks][21]
11. [Recurrent networks and Natural Language Processing][22]

### Guest lectures

12. [Deep Learning in the Real World][23] (Soumith Chintala, Facebook)
13. [From PyTorch To TensorFlow][24] (Andreas Steiner, Google)
14. [QuickDraw End2end using TensorFlow][25] (Andreas Steiner, Google)

## Lecture 1 (Feb 21, 2018) – Introduction and tensors

![Thumbnail made from a slide][26]

What is deep learning, some history, what are the current applications. torch.Tensor, linear regression. 

## Lecture 2 (Feb 28, 2018) – Machine learning fundamentals

![Thumbnail made from a slide][27]

Empirical risk minimization, capacity, bias-variance dilemma, polynomial regression, k-means and PCA.

## Lecture 3 (Mar 07, 2018) – Multi-layer perceptrons

![Thumbnail made from a slide][28]

Linear classifiers, perceptron, linear separability and feature extraction, Multi-Layer Perceptron, gradient descent, back-propagation.

## Lecture 4 (Mar 14, 2018) – Convolutional networks and autograd

![Thumbnail made from a slide][29]

Generalized acyclic graph networks, torch.autograd, batch processing, convolutional layers and pooling, torch.nn.Module.

## Lecture 5 (Mar 21, 2018) – Optimization

![Thumbnail made from a slide][30]

Cross-entropy, L1 and L2 penalty. Vanishing gradient, weight initialization, Xavier's rule, loss monitoring. torch.autograd.Function.

## Lecture 6 (Mar 28, 2018) – Going deeper

![Thumbnail made from a slide][31]

Theoretical advantages of depth, rectifiers, drop-out, batch normalization, residual networks, advanced weight initialization. GPUs and torch.cuda.

## No lecture (Apr 4, 2018) – Easter break

## Lecture 7 (Apr 11, 2018) – Computer vision

![Thumbnail made from a slide][32]

Deep networks for image classification (AlexNet, VGGNet), object detection (YOLO), and semantic segmentation (FCN). Data-loaders, neuro-surgery, and fine-tuning.

## Lecture 8 (Apr 18, 2018) – Under the hood

![Thumbnail made from a slide][33]

Visualizing filters and activations, smoothgrad, deconvolution, guided back-propagation. Optimizing samples from scratch, adversarial examples. Dilated convolutions.

## Lecture 9 (Apr 25, 2018) – Autoencoders and generative models

![Thumbnail made from a slide][34]

Transposed convolution layers, autoencoders, variational autoencoders, non volume-preserving networks.

## Lecture 10 (May 2, 2018) – Generative Adversarial Networks

![Thumbnail made from a slide][35]

GAN, Wasserstein GAN, Deep Convolutional GAN, Image-to-Image translations, model persistence.

## Lecture 11 (May 9, 2018) – Recurrent networks and Natural Language Processing

![Thumbnail made from a slide][36]

Back-propagation through time, gating, LSTM, GRU. Word embeddings, sentence-to-sentence translation.

## Lecture 12 (May 16, 2018) – Deep Learning in the Real World

**Guest speaker:** [Soumith Chintala][37] (Facebook)

![Speaker in front of the audience][38]

Large data-sets and models, effectively parallelizing on GPUs, pythonless deployment. torch.distributed, ONNX, exporting to Caffe2.

## Lecture 13 (May 23, 2018) – From PyTorch To TensorFlow

**Guest speaker:** Andreas Steiner (Google)

![Speaker in front of the audience][39]

ML research at Google, mobile ML application example, Low-level TensorFlow Python API (Graph, Session, shapes, variables).

**Restricted to enrolled students:**

## Lecture 14 (May 30, 2018) – QuickDraw End2end using TF

**Guest speaker:** Andreas Steiner (Google)

Transforming QuickDraw data, estimator and experiment interfaces, CNN/RNN classifiers, CloudML. 

**Restricted to enrolled students:**

## Pre-requisites

* Linear algebra (vector and Euclidean spaces),
* differential calculus (Jacobian, Hessian, chain rule),
* Python,
* basics in probabilities and statistics (discrete and continuous distributions, law of large numbers, conditional probabilities, Bayes, PCA),
* basics in optimization (notion of minima, gradient descent),
* basics in algorithmic (computational costs),
* basics in signal processing (Fourier transform, wavelets).

## Documentation

You may have to look at the python 3, jupyter, and PyTorch documentations at

## Grading

The final grade will be 25% for each of the two [mini-projects][40] grade, and 50% for the written exam during the exam session.

## Practical session prologue

Helper python prologue for the practical sessions: [dlc_practical_prologue.py][41]

### Argument parsing

This prologue parses command-line arguments as follows
    
    
    usage: dummy.py [-h] [--full] [--tiny] [--force_cpu] [--seed SEED]
                    [--cifar] [--data_dir DATA_DIR]
    
    DLC prologue file for practical sessions.
    
    optional arguments:
      -h, --help           show this help message and exit
      --full               Use the full set, can take ages (default
                           False)
      --tiny               Use a very small set for quick checks
                           (default False)
      --force_cpu          Keep tensors on the CPU, even if cuda is
                           available (default False)
      --seed SEED          Random seed (default 0, < 0 is no seeding)
      --cifar              Use the CIFAR data-set and not MNIST
                           (default False)
      --data_dir DATA_DIR  Where are the PyTorch data located (default
                           $PYTORCH_DATA_DIR or './data')
    

It sets the default Tensor to torch.cuda.FloatTensor if cuda is available (and \--force_cpu is not set).

### Loading data

The prologue provides the function
    
    
    load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True)
    

which downloads the data when required, reshapes the images to 1d vectors if flatten is True, narrows to a small subset of samples if \--full is not selected, moves the Tensors to the GPU if cuda is available (and \--force_cpu is not selected).

It returns a tuple of four tensors: train_data, train_target, test_data, and test_target.

If cifar is True, the data-base used is CIFAR10, if it is False, MNIST is used, if it is None, the argument \--cifar is taken into account.

If one_hot_labels is True, the targets are converted to 2d torch.Tensor with as many columns as there are classes, and -1 everywhere except the coefficients [n, y_n], equal to 1.

If normalize is True, the data tensors are normalized according to the mean and variance of the training one.

If flatten is True, the data tensors are flattened into 2d tensors of dimension N × D, discarding the image structure of the samples. Otherwise they are 4d tensors of dimension N × C × H × W.

### Minimal example
    
    
    import dlc_practical_prologue as prologue
    
    train_input, train_target, test_input, test_target = prologue.load_data()
    
    print('train_input', train_input.size(), 'train_target', train_target.size())
    print('test_input', test_input.size(), 'test_target', test_target.size())
    

prints
    
    
    data_dir ./data
    * Using MNIST
    ** Reduce the data-set (use --full for the full thing)
    ** Use 1000 train and 1000 test samples
    train_input torch.Size([1000, 784]) train_target torch.Size([1000])
    test_input torch.Size([1000, 784]) test_target torch.Size([1000])
    

A Virtual Machine (VM) is a software that simulates a complete computer. The one we provide here includes a Linux operating system and all the tools needed to use PyTorch from a web browser (firefox or chrome).

## Installation at the EPFL

It is already installed on the machines in room CM1 103 for the exercise sessions. You can start it with Windows Start menu → All programs → VirtualBox → VB_DEEP_LEARNING. 

## Installation of the VM on your own computer

If you want to use your own machine, first download and install: [Oracle's VirtualBox][42] then download the file: [Virtual machine OVA package (large file ~3.6Gb)][43] and open it in VirtualBox with File → Import Appliance.

You should now see an entry in the list of VMs. The first time it starts, it provides a menu to choose the keyboard layout you want to use (you can force the configuration later by passing forcekbd to the kernel through GRUB).

**If the VM does not start and VirtualBox complains that the VT-x is not enabled, you have to activate the virtualization capabilities of your Intel CPU in the BIOS of your computer.**

## Using the VM

The VM automatically starts a [JupyterLab][44] on port 8888 and exports that port to the host. This means that you can access this JupyterLab with a web browser on the machine running VirtualBox at:  and use python notebooks, view files, start terminals, and edit source files. Typing !bye in a notebook or bye in a terminal will shutdown the VM.

You can run a terminal and a text editor from inside the Jupyter notebook for exercises that require more than the notebook itself. Source files can be executed by running in a terminal the python command with the source file name as argument. Both can be done from the main Jupyter window with:

* New → Text File to create the source code, or selecting the file and clicking Edit to edit an existing one.
* New → Terminal to start a shell from which you can run python.

**Files saved in the VM are erased when the VM is re-installed, which happens for each session on the EPFL machines. So you should download files you want to keep from the jupyter notebook to your account and re-upload them later when you need them.**

This VM also exports an ssh port to the port 2022 on the host, which allows to log in with standard ssh clients on Linux and OSX, and with applications such as [PuTTY][45] on Windows. The default login is 'dave' and password 'dummy', same password for the root.

## Remarks

Note that performance for computation will not be as good as if you [install PyTorch][46] natively on your machine (which is possible only on Linux and OSX for versions < 0.4.0). In particular, the VM does not take advantage of a GPU if you have one.

**Finally, please also note that this VM is configured in a convenient but highly non-secured manner, with easy to guess passwords, including for the root, and network-accessible non-protected Jupyter notebooks.**

This VM is built on a [Linux][47] [Debian 9.3 "stretch",][48] with [miniconda,][49] [PyTorch 0.3.1,][46] [TensorFlow 1.4.1,][50] [MNIST,][51] [CIFAR10,][52] and many Python utility packages installed.

Here are the two mandatory mini-projects:

The pdf files and videos on this page are licensed under the [Creative Commons BY-NC-SA 4.0 International License.][53]

More simply: I am okay with this material being used for regular academic teaching, but definitely not for a book / youtube loaded with ads / whatever monetization model I am not aware of.

[1]: https://documents.epfl.ch/www-pics/logo_idiap.png
[2]: http://www.idiap.ch/
[3]: https://documents.epfl.ch/www-pics/logo_epfl.png
[4]: http://www.epfl.ch/
[5]: http://www.epfl.ch
[6]: http://edu.epfl.ch/coursebook/en/deep-learning-EE-559
[7]: http://www.idiap.ch/~fleuret/
[8]: http://pytorch.org
[9]: https://fleuret.org/ee559/
[10]: https://documents.epfl.ch#vm
[11]: https://documents.epfl.ch#prologue
[12]: https://documents.epfl.ch#course-1
[13]: https://documents.epfl.ch#course-2
[14]: https://documents.epfl.ch#course-3
[15]: https://documents.epfl.ch#course-4
[16]: https://documents.epfl.ch#course-5
[17]: https://documents.epfl.ch#course-6
[18]: https://documents.epfl.ch#course-7
[19]: https://documents.epfl.ch#course-8
[20]: https://documents.epfl.ch#course-9
[21]: https://documents.epfl.ch#course-10
[22]: https://documents.epfl.ch#course-11
[23]: https://documents.epfl.ch#course-12
[24]: https://documents.epfl.ch#course-13
[25]: https://documents.epfl.ch#course-14
[26]: https://documents.epfl.ch/www-pics/thumb-1.png
[27]: https://documents.epfl.ch/www-pics/thumb-2.png
[28]: https://documents.epfl.ch/www-pics/thumb-3.png
[29]: https://documents.epfl.ch/www-pics/thumb-4.png
[30]: https://documents.epfl.ch/www-pics/thumb-5.png
[31]: https://documents.epfl.ch/www-pics/thumb-6.png
[32]: https://documents.epfl.ch/www-pics/thumb-7.png
[33]: https://documents.epfl.ch/www-pics/thumb-8.png
[34]: https://documents.epfl.ch/www-pics/thumb-9.png
[35]: https://documents.epfl.ch/www-pics/thumb-10.png
[36]: https://documents.epfl.ch/www-pics/thumb-11.png
[37]: https://research.fb.com/people/chintala-soumith/
[38]: https://documents.epfl.ch/www-pics/thumb-12.png
[39]: https://documents.epfl.ch/www-pics/thumb-13.png
[40]: https://documents.epfl.ch#mini-projects
[41]: https://documents.epfl.ch/dlc_practical_prologue.py
[42]: https://www.virtualbox.org/wiki/Downloads
[43]: https://documents.epfl.ch/users/f/fl/fleuret/www/ova/Deep%20Learning%20VM%200.2.ova
[44]: http://jupyter.org/
[45]: https://www.putty.org/
[46]: http://pytorch.org/
[47]: https://www.linuxfoundation.org/
[48]: https://www.debian.org/
[49]: https://conda.io/miniconda.html
[50]: https://www.tensorflow.org/
[51]: http://yann.lecun.com/exdb/mnist/
[52]: https://www.cs.toronto.edu/~kriz/cifar.html
[53]: https://documents.epfl.ch/by-nc-sa-4.0.txt

  