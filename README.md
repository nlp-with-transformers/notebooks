# Transformers Notebooks

This repository contains the example code from our O'Reilly book [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/):

<img alt="book-cover" height=200 src="images/book_cover.jpg" id="book-cover"/>

## Getting started

You can run these notebooks on platforms like Google Colab or your local machine. Note that most chapters require a GPU to run in a reasonable amount of time, so we recommend Colab as it comes pre-installed with CUDA.

Nowadays, the GPUs on Colab tend to be K80s (which have limited memory), so alternative platforms we recommend include:

* [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)
* [Gradient Notebooks](https://gradient.run/notebooks)
* [SageMaker Studio Lab](https://studiolab.sagemaker.aws/)

These platforms tend to provide more performant GPUs like P100s, all for free!

### Running on Google Colab

To run these notebooks on Colab, just click on the <!--<badge>--><a href="https://colab.research.google.com/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><!--</badge>--> button at the top of each notebook.

### Running on your machine

To run the notebooks on your own machine, first clone the repository:

```
git clone https://github.com/nlp-with-transformers/notebooks
cd notebooks-test
```

Next, you'll need to install a few packages that depend on your operating system and hardware:

* [PyTorch](https://pytorch.org/get-started/locally/)
* [TensorFlow](https://www.tensorflow.org/install/)) (optional, since only used in a few chapters)
* [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) (only used in Chapter 11)
* [librosa](https://librosa.org/) (only used in Chapter 11)
* [libsndfile](http://www.mega-nerd.com/libsndfile/) (only used in Chapter 11)

Once you have install the above requirements, create a virtual environment and install the remaining Python dependencies:

```
conda create -n book python=3.8 -y && conda activate book
from install import *
install_requirements()
# Use the following to run Chapter 7
# install_requirements(is_chapter7)
```
