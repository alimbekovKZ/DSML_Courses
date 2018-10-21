
[Source](https://fleuret.org/ee559/ "Permalink to Deep Learning Course (Spring 2019)")

# Deep Learning Course (Spring 2019)

## [Old Version With Video - Deep Learning Course (Spring 2018)][139]

You can find here [slides][5] and a [virtual machine][6] for the course [EE-559 "Deep Learning"][7], taught by [François Fleuret][8] at [EPFL][9]. This course covers the main deep learning tools and theoretical results, with examples in the [PyTorch][10] framework. **Note that these slides are still work in progress at the moment.**

[Last year's version][11] provides handouts and 16h of voice-overs, but is structured slightly differently, and was developed for [PyTorch 0.3.1,][12] which differs substantially from [PyTorch 0.4.x.][13]

Thanks to Adam Paszke, Alexandre Nanchen, Xavier Glorot, Andreas Steiner, Matus Telgarsky, Diederik Kingma, Nikolaos Pappas, Soumith Chintala, and Shaojie Bai for their answers or comments.

## Slides

The slide pdfs are the ones I use during the lectures. They are in landscape mode and include overlays and font coloring to facilitate the presentation. The handout pdfs are compiled without these fancy effects and with two slides per page in portrait mode to be more convenient for off-line reading and note-taking.

1. Introduction. 
    1. From artificial neural networks to deep learning. ([slides][14], [handout][15] – 21 slides) 
    2. Current applications and success. ([slides][16], [handout][17] – 22 slides) 
    3. What is really happening? ([slides][18], [handout][19] – 13 slides) 
    4. Tensor basics and linear regression. ([slides][20], [handout][21] – 12 slides) 
    5. High dimension tensors. ([slides][22], [handout][23] – 14 slides) 
    6. Tensor internals. ([slides][24], [handout][25] – 5 slides) 
2. Machine learning fundamentals. 
    1. Loss and risk. ([slides][26], [handout][27] – 15 slides) 
    2. Over and under fitting. ([slides][28], [handout][29] – 24 slides) 
    3. Bias-variance dilemma. ([slides][30], [handout][31] – 10 slides) 
    4. Proper evaluation protocols. ([slides][32], [handout][33] – 6 slides) 
    5. Basic clustering and embeddings. ([slides][34], [handout][35] – 19 slides) 
3. Multi-layer perceptron and back-propagation. 
    1. The perceptron. ([slides][36], [handout][37] – 16 slides) 
    2. Probabilistic interpretation of the linear classifier. ([slides][38], [handout][39] – 8 slides) 
    3. Limitation of the linear classifier, feature design. ([slides][40], [handout][41] – 10 slides) 
    4. Multi-Layer Perceptrons. ([slides][42], [handout][43] – 9 slides) 
    5. Gradient descent. ([slides][44], [handout][45] – 13 slides) 
    6. Back-propagation. ([slides][46], [handout][47] – 11 slides) 
4. Graphs of operators, autograd, and convolutional layers. 
    1. DAG networks. ([slides][48], [handout][49] – 11 slides) 
    2. Autograd. ([slides][50], [handout][51] – 16 slides) 
    3. PyTorch modules and batch processing. ([slides][52], [handout][53] – 14 slides) 
    4. Convolutions. ([slides][54], [handout][55] – 23 slides) 
    5. Pooling. ([slides][56], [handout][57] – 7 slides) 
    6. Writing a PyTorch module. ([slides][58], [handout][59] – 10 slides) 
5. Initialization and optimization. 
    1. Cross-entropy loss. ([slides][60], [handout][61] – 9 slides) 
    2. Stochastic gradient descent. ([slides][62], [handout][63] – 17 slides) 
    3. PyTorch optimizers. ([slides][64], [handout][65] – 7 slides) 
    4. $L_2$ and $L_1$ penalties. ([slides][66], [handout][67] – 10 slides) 
    5. Parameter initialization. ([slides][68], [handout][69] – 22 slides) 
    6. Architecture choice and training protocol. ([slides][70], [handout][71] – 9 slides) 
    7. Writing an autograd function. ([slides][72], [handout][73] – 7 slides) 
6. Going deeper. 
    1. Benefits of depth. ([slides][74], [handout][75] – 9 slides) 
    2. Rectifiers. ([slides][76], [handout][77] – 7 slides) 
    3. Dropout. ([slides][78], [handout][79] – 12 slides) 
    4. Batch normalization. ([slides][80], [handout][81] – 15 slides) 
    5. Residual networks. ([slides][82], [handout][83] – 21 slides) 
    6. Using GPUs. ([slides][84], [handout][85] – 15 slides) 
7. Computer vision. 
    1. Computer vision tasks. ([slides][86], [handout][87] – 15 slides) 
    2. Networks for image classification. ([slides][88], [handout][89] – 36 slides) 
    3. Networks for object detection. ([slides][90], [handout][91] – 15 slides) 
    4. Networks for semantic segmentation. ([slides][92], [handout][93] – 8 slides) 
    5. DataLoader and neuro-surgery. ([slides][94], [handout][95] – 13 slides) 
8. Under the hood. 
    1. Looking at parameters. ([slides][96], [handout][97] – 11 slides) 
    2. Looking at activations. ([slides][98], [handout][99] – 21 slides) 
    3. Visualizing the processing in the input. ([slides][100], [handout][101] – 26 slides) 
    4. Optimizing inputs. ([slides][102], [handout][103] – 25 slides) 
9. Auto-encoders and generative models. 
    1. Transposed convolutions. ([slides][104], [handout][105] – 14 slides) 
    2. Autoencoders. ([slides][106], [handout][107] – 20 slides) 
    3. Denoising and variational autoencoders. ([slides][108], [handout][109] – 24 slides) 
    4. Non-volume preserving networks. ([slides][110], [handout][111] – 24 slides) 
10. Generative adversarial models. 
    1. Generative Adversarial Networks. ([slides][112], [handout][113] – 29 slides) 
    2. Wasserstein GAN. ([slides][114], [handout][115] – 16 slides) 
    3. Conditional GAN and image translation. ([slides][116], [handout][117] – 27 slides) 
    4. Model persistence and checkpoints. ([slides][118], [handout][119] – 9 slides) 
11. Recurrent models and NLP. 
    1. Recurrent Neural Networks. ([slides][120], [handout][121] – 23 slides) 
    2. LSTM and GRU. ([slides][122], [handout][123] – 17 slides) 
    3. Word embeddings and translation. ([slides][124], [handout][125] – 31 slides) 

## Practicals

## Pre-requisites

* Linear algebra (vector and Euclidean spaces),
* differential calculus (Jacobian, Hessian, chain rule),
* Python programming,
* basics in probabilities and statistics (discrete and continuous distributions, law of large numbers, conditional probabilities, Bayes, PCA),
* basics in optimization (notion of minima, gradient descent),
* basics in algorithmic (computational costs),
* basics in signal processing (Fourier transform, wavelets).

## Documentation

You may have to look at the python 3, jupyter, and PyTorch documentations at

## Practical session prologue

Helper python prologue for the practical sessions: [dlc_practical_prologue.py][126]

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

### Installation

First download and install: [Oracle's VirtualBox][127] then download the file: [Virtual machine OVA package (large file ~2.5Gb)][128] and open it in VirtualBox with File → Import Appliance.

You should now see an entry in the list of VMs. The first time it starts, it provides a menu to choose the keyboard layout you want to use (you can force the configuration later by passing forcekbd to the kernel through GRUB).

**If the VM does not start and VirtualBox complains that the VT-x is not enabled, you have to activate the virtualization capabilities of your Intel CPU in the BIOS of your computer.**

### Using the VM

The VM automatically starts a [JupyterLab][129] on port 8888 and exports that port to the host. This means that you can access this JupyterLab with a web browser on the machine running VirtualBox at:  and use python notebooks, view files, start terminals, and edit source files. Typing !bye in a notebook or bye in a terminal will shutdown the VM.

You can run a terminal and a text editor from inside the Jupyter notebook for exercises that require more than the notebook itself. Source files can be executed by running in a terminal the python command with the source file name as argument. Both can be done from the main Jupyter window with:

* New → Text File to create the source code, or selecting the file and clicking Edit to edit an existing one.
* New → Terminal to start a shell from which you can run python.

**Files saved in the VM are erased when the VM is re-installed, which happens for each session on the EPFL machines. So you should download files you want to keep from the jupyter notebook to your account and re-upload them later when you need them.**

This VM also exports an ssh port to the port 2022 on the host, which allows to log in with standard ssh clients on Linux and OSX, and with applications such as [PuTTY][130] on Windows. The default login is 'dave' and password 'dummy', same password for the root.

### Remarks

Note that performance for computation will not be as good as if you [install PyTorch][131] natively on your machine. In particular, the VM does not take advantage of a GPU if you have one.

**Finally, please also note that this VM is configured in a convenient but highly non-secured manner, with easy to guess passwords, including for the root, and network-accessible non-protected Jupyter notebooks.**

This VM is built on a [Linux][132] [Debian 9.3 "stretch",][133] with [miniconda,][134] [PyTorch 0.4.1,][131] [TensorFlow 1.4.1,][135] [MNIST,][136] [CIFAR10,][137] and many Python utility packages installed.

The materials on this page are licensed under the [Creative Commons BY-NC-SA 4.0 International License.][138]

More simply: I am okay with this material being used for regular academic teaching, but definitely not for a book / youtube loaded with ads / whatever monetization model I am not aware of.

[1]: https://fleuret.org/www-pics/logo_idiap.png
[2]: http://www.idiap.ch/
[3]: https://fleuret.org/www-pics/logo_epfl.png
[4]: http://www.epfl.ch/
[5]: https://fleuret.org#slides
[6]: https://fleuret.org#vm
[7]: http://edu.epfl.ch/coursebook/en/deep-learning-EE-559
[8]: http://www.idiap.ch/~fleuret/
[9]: http://www.epfl.ch
[10]: http://pytorch.org
[11]: https://fleuret.org/dlc
[12]: https://pytorch.org/docs/0.3.1/
[13]: https://pytorch.org/docs/0.4.0/
[14]: https://fleuret.org/ee559-slides-1-1-from-anns-to-deep-learning.pdf
[15]: https://fleuret.org/ee559-handout-1-1-from-anns-to-deep-learning.pdf
[16]: https://fleuret.org/ee559-slides-1-2-current-success.pdf
[17]: https://fleuret.org/ee559-handout-1-2-current-success.pdf
[18]: https://fleuret.org/ee559-slides-1-3-what-is-happening.pdf
[19]: https://fleuret.org/ee559-handout-1-3-what-is-happening.pdf
[20]: https://fleuret.org/ee559-slides-1-4-tensors-and-linear-regression.pdf
[21]: https://fleuret.org/ee559-handout-1-4-tensors-and-linear-regression.pdf
[22]: https://fleuret.org/ee559-slides-1-5-high-dimension-tensors.pdf
[23]: https://fleuret.org/ee559-handout-1-5-high-dimension-tensors.pdf
[24]: https://fleuret.org/ee559-slides-1-6-tensor-internals.pdf
[25]: https://fleuret.org/ee559-handout-1-6-tensor-internals.pdf
[26]: https://fleuret.org/ee559-slides-2-1-loss-and-risk.pdf
[27]: https://fleuret.org/ee559-handout-2-1-loss-and-risk.pdf
[28]: https://fleuret.org/ee559-slides-2-2-overfitting.pdf
[29]: https://fleuret.org/ee559-handout-2-2-overfitting.pdf
[30]: https://fleuret.org/ee559-slides-2-3-bias-variance-dilemma.pdf
[31]: https://fleuret.org/ee559-handout-2-3-bias-variance-dilemma.pdf
[32]: https://fleuret.org/ee559-slides-2-4-evaluation-protocols.pdf
[33]: https://fleuret.org/ee559-handout-2-4-evaluation-protocols.pdf
[34]: https://fleuret.org/ee559-slides-2-5-basic-embeddings.pdf
[35]: https://fleuret.org/ee559-handout-2-5-basic-embeddings.pdf
[36]: https://fleuret.org/ee559-slides-3-1-perceptron.pdf
[37]: https://fleuret.org/ee559-handout-3-1-perceptron.pdf
[38]: https://fleuret.org/ee559-slides-3-2-LDA.pdf
[39]: https://fleuret.org/ee559-handout-3-2-LDA.pdf
[40]: https://fleuret.org/ee559-slides-3-3-features.pdf
[41]: https://fleuret.org/ee559-handout-3-3-features.pdf
[42]: https://fleuret.org/ee559-slides-3-4-MLP.pdf
[43]: https://fleuret.org/ee559-handout-3-4-MLP.pdf
[44]: https://fleuret.org/ee559-slides-3-5-gradient-descent.pdf
[45]: https://fleuret.org/ee559-handout-3-5-gradient-descent.pdf
[46]: https://fleuret.org/ee559-slides-3-6-backprop.pdf
[47]: https://fleuret.org/ee559-handout-3-6-backprop.pdf
[48]: https://fleuret.org/ee559-slides-4-1-DAG-networks.pdf
[49]: https://fleuret.org/ee559-handout-4-1-DAG-networks.pdf
[50]: https://fleuret.org/ee559-slides-4-2-autograd.pdf
[51]: https://fleuret.org/ee559-handout-4-2-autograd.pdf
[52]: https://fleuret.org/ee559-slides-4-3-modules-and-batch-processing.pdf
[53]: https://fleuret.org/ee559-handout-4-3-modules-and-batch-processing.pdf
[54]: https://fleuret.org/ee559-slides-4-4-convolutions.pdf
[55]: https://fleuret.org/ee559-handout-4-4-convolutions.pdf
[56]: https://fleuret.org/ee559-slides-4-5-pooling.pdf
[57]: https://fleuret.org/ee559-handout-4-5-pooling.pdf
[58]: https://fleuret.org/ee559-slides-4-6-writing-a-module.pdf
[59]: https://fleuret.org/ee559-handout-4-6-writing-a-module.pdf
[60]: https://fleuret.org/ee559-slides-5-1-cross-entropy-loss.pdf
[61]: https://fleuret.org/ee559-handout-5-1-cross-entropy-loss.pdf
[62]: https://fleuret.org/ee559-slides-5-2-SGD.pdf
[63]: https://fleuret.org/ee559-handout-5-2-SGD.pdf
[64]: https://fleuret.org/ee559-slides-5-3-optim.pdf
[65]: https://fleuret.org/ee559-handout-5-3-optim.pdf
[66]: https://fleuret.org/ee559-slides-5-4-l2-l1-penalties.pdf
[67]: https://fleuret.org/ee559-handout-5-4-l2-l1-penalties.pdf
[68]: https://fleuret.org/ee559-slides-5-5-initialization.pdf
[69]: https://fleuret.org/ee559-handout-5-5-initialization.pdf
[70]: https://fleuret.org/ee559-slides-5-6-architecture-and-training.pdf
[71]: https://fleuret.org/ee559-handout-5-6-architecture-and-training.pdf
[72]: https://fleuret.org/ee559-slides-5-7-writing-an-autograd-function.pdf
[73]: https://fleuret.org/ee559-handout-5-7-writing-an-autograd-function.pdf
[74]: https://fleuret.org/ee559-slides-6-1-benefits-of-depth.pdf
[75]: https://fleuret.org/ee559-handout-6-1-benefits-of-depth.pdf
[76]: https://fleuret.org/ee559-slides-6-2-rectifiers.pdf
[77]: https://fleuret.org/ee559-handout-6-2-rectifiers.pdf
[78]: https://fleuret.org/ee559-slides-6-3-dropout.pdf
[79]: https://fleuret.org/ee559-handout-6-3-dropout.pdf
[80]: https://fleuret.org/ee559-slides-6-4-batch-normalization.pdf
[81]: https://fleuret.org/ee559-handout-6-4-batch-normalization.pdf
[82]: https://fleuret.org/ee559-slides-6-5-residual-networks.pdf
[83]: https://fleuret.org/ee559-handout-6-5-residual-networks.pdf
[84]: https://fleuret.org/ee559-slides-6-6-using-GPUs.pdf
[85]: https://fleuret.org/ee559-handout-6-6-using-GPUs.pdf
[86]: https://fleuret.org/ee559-slides-7-1-CV-tasks.pdf
[87]: https://fleuret.org/ee559-handout-7-1-CV-tasks.pdf
[88]: https://fleuret.org/ee559-slides-7-2-image-classification.pdf
[89]: https://fleuret.org/ee559-handout-7-2-image-classification.pdf
[90]: https://fleuret.org/ee559-slides-7-3-object-detection.pdf
[91]: https://fleuret.org/ee559-handout-7-3-object-detection.pdf
[92]: https://fleuret.org/ee559-slides-7-4-segmentation.pdf
[93]: https://fleuret.org/ee559-handout-7-4-segmentation.pdf
[94]: https://fleuret.org/ee559-slides-7-5-dataloader-and-surgery.pdf
[95]: https://fleuret.org/ee559-handout-7-5-dataloader-and-surgery.pdf
[96]: https://fleuret.org/ee559-slides-8-1-looking-at-parameters.pdf
[97]: https://fleuret.org/ee559-handout-8-1-looking-at-parameters.pdf
[98]: https://fleuret.org/ee559-slides-8-2-looking-at-activations.pdf
[99]: https://fleuret.org/ee559-handout-8-2-looking-at-activations.pdf
[100]: https://fleuret.org/ee559-slides-8-3-visualizing-in-input.pdf
[101]: https://fleuret.org/ee559-handout-8-3-visualizing-in-input.pdf
[102]: https://fleuret.org/ee559-slides-8-4-optimizing-inputs.pdf
[103]: https://fleuret.org/ee559-handout-8-4-optimizing-inputs.pdf
[104]: https://fleuret.org/ee559-slides-9-1-transposed-convolutions.pdf
[105]: https://fleuret.org/ee559-handout-9-1-transposed-convolutions.pdf
[106]: https://fleuret.org/ee559-slides-9-2-autoencoders.pdf
[107]: https://fleuret.org/ee559-handout-9-2-autoencoders.pdf
[108]: https://fleuret.org/ee559-slides-9-3-denoising-and-variational-autoencoders.pdf
[109]: https://fleuret.org/ee559-handout-9-3-denoising-and-variational-autoencoders.pdf
[110]: https://fleuret.org/ee559-slides-9-4-NVP.pdf
[111]: https://fleuret.org/ee559-handout-9-4-NVP.pdf
[112]: https://fleuret.org/ee559-slides-10-1-GAN.pdf
[113]: https://fleuret.org/ee559-handout-10-1-GAN.pdf
[114]: https://fleuret.org/ee559-slides-10-2-Wasserstein-GAN.pdf
[115]: https://fleuret.org/ee559-handout-10-2-Wasserstein-GAN.pdf
[116]: https://fleuret.org/ee559-slides-10-3-conditional-GAN.pdf
[117]: https://fleuret.org/ee559-handout-10-3-conditional-GAN.pdf
[118]: https://fleuret.org/ee559-slides-10-4-persistence.pdf
[119]: https://fleuret.org/ee559-handout-10-4-persistence.pdf
[120]: https://fleuret.org/ee559-slides-11-1-RNN-basics.pdf
[121]: https://fleuret.org/ee559-handout-11-1-RNN-basics.pdf
[122]: https://fleuret.org/ee559-slides-11-2-LSTM-and-GRU.pdf
[123]: https://fleuret.org/ee559-handout-11-2-LSTM-and-GRU.pdf
[124]: https://fleuret.org/ee559-slides-11-3-word-embeddings-and-translation.pdf
[125]: https://fleuret.org/ee559-handout-11-3-word-embeddings-and-translation.pdf
[126]: https://fleuret.org/dlc_practical_prologue.py
[127]: https://www.virtualbox.org/wiki/Downloads
[128]: https://drive.switch.ch/index.php/s/R5jBbjz2GQG4vhh/download?path=%2F&files=ee559-vm-09102018.ova
[129]: http://jupyter.org/
[130]: https://www.putty.org/
[131]: http://pytorch.org/
[132]: https://www.linuxfoundation.org/
[133]: https://www.debian.org/
[134]: https://conda.io/miniconda.html
[135]: https://www.tensorflow.org/
[136]: http://yann.lecun.com/exdb/mnist/
[137]: https://www.cs.toronto.edu/~kriz/cifar.html
[138]: https://fleuret.org/by-nc-sa-4.0.txt
[139]: https://github.com/alimbekovKZ/DSML_Courses/blob/master/EE-559%20%E2%80%93%20Deep%20Learning/README_OLD.md
  
