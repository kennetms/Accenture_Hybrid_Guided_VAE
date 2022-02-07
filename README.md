# Accenture_Hybrid_Guided_VAE

The [Hybrid Guided VAE](https://arxiv.org/abs/2104.00165) is a method for . [Accenture Labs](https://www.accenture.com/us-en/about/accenture-labs-index) created the Hybrid Guided VAE in collaboration with the [UCI NMI Lab](https://nmi-lab.org/). By open-sourcing the components that we used to enable training SNN models with our method, we hope to encourage adoption to other datasets and problem domains and collaboration for improvements to the methods.

## Table of Contents

+ [**Process Overview**](#process-overview)
+ [**Getting Data**](#getting-data)
+ [**Training Models**](#training-models)
+ [**Trained models**](#trained-models)
+ [**Training with SLAYER**](#training-with-slayer)
+ [**Licensing**](#licensing)
+ [**How to Contribute**](#how-to-contribute)
+ [**How to Cite**](#how-to-cite)
+ [**Contacts**](#contacts)

## Process Overview

​
It is recommended to create a python3 virtual environment before executing further steps.
This is how you do it:
​

```bash
$ python3 -m venv hgvae
source hgvae/bin/activate
```

​
To install the required libraries used to help train and run Hybrid Guided VAE models, run the following:
​
```bash
$ pip install -r requirements.txt
```

Now you will need to get the data and follow instructions for training models in the following steps.

## Getting Data

**DVSGestures**
​
To install the DVSGestures dataset to train and run Hybrid Guided VAE models, run the following:
​

```bash
$ cd data
$ wget https://www.dropbox.com/s/3v0t4hn0c9asior/dvs_zipped.zip\?dl=0
$ unzip 'dvs_zipped.zip?dl=0'
$ cd ..
```

**N-MNIST** 
​
The torchenuromorphic library will automatically install the N-MNIST dataset if it is not on your local machine 
when you try to train or run an N-MNIST model.
​

## Training Models

There are

To train and evaluate the DVSGestures run the following line:
```bash
$ cd dvs_gestures
$ python train_gestures.py
```

To train and evaluate the DVSGestures guided on lighting:
```bash
$ cd dvs_gesture_lighting
$ python train_lights.py
```
To train and evaluate on the NMNIST data:
```bash
$ cd nmnist
$ python train_nmnist.py
```
Note: the NMNIST dataset will be downloaded automatically when this is run.

To train a model with SLAYER-Loihi that is compatible with Loihi 1.0 run the .ipynb notebook in the snn_loihi_example folder.
```bash
$ cd slayer_loihi_example
```
​
Loihi_Simulator_training.ipynb
​

Models typically take several hours to train, with intermediate results and models stored in the logs/ directory.

## Trained models
Example models with their parameters can be found in the subfolders named
​
example_model
​


Checkpoints with saved models are placed in the logs directory.
Intermediate results can be viewed with tensorboard.

For example:
```bash
$ tensorboard --logdir logs/train_lights/default/Mar23_20-57-42_ripper --port 6007 --bind_all
```

The files in the parameters/ directory can be changed to load a model from a saved checkpoint.

For example, to resume a model from a checkpoint edit the line
​
resume_from: None
​

to the checkpoints folder of the model you want to resume from. 
For example, if you wanted to use the provided DVSGestures model:
​
resume_from: ../dvs_gestures/example_model/checkpoints/
​

To train a model from scratch, change the line to:
​
resume_from: None
​

## Licensing
These assets are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

## How to Contribute
We welcome community contributions, especially for new models, improvements to the post-processing, and documentation.

If you'd like to create and share new models, models with new datasets, or improvements, you can do so by opening up a pull request.  

## How to Cite

If you use or adopt the models, code, or methods presented here please cite our work as follows:

@misc{stewart2021gesture,
      title={Gesture Similarity Analysis on Event Data Using a Hybrid Guided Variational Auto Encoder}, 
      author={Kenneth Stewart and Andreea Danielescu and Lazar Supic and Timothy Shea and Emre Neftci},
      year={2021},
      eprint={2104.00165},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}


## Contacts

Andreea Danielescu\
​*Future Technologies, Accenture Labs*\
[andreea.danielescu@accenture.com](mailto:@accenture.com?subject=[GitHub])

​Kenneth Stewart\
​*PhD Candidate, University of California, Irvine*\
​[kennetms@uci.edu](mailto:kennetms@uci.edu?subject=[GitHub])