# Project Overview
This project aims to classify brain synapses into excitatory and inhibitory types using 2D electron microscopy (EM) patches and 3D image cubes extracted from the MICrONS dataset. Our best 2D model was the ResNet-50 model @ 76.89% accuracy. Our best 3D model was...


# Setup
#### 1. Clone the repository (or click download):
```
git clone https://github.com/<your_repo>.git
cd <your_repo>
```

#### 2. Install dependencies (or install manually on Anaconda Prompt)
```
pip install -r requirements.txt
```

# Running the Demos

#### 1. Download the 2D model and 2D dataset and place them in the checkpoints and data folder respectively:

###### Google Drive Links

2D ResNet-50 Model w/ 76.89% Accuracy:
https://drive.google.com/file/d/1CUZ2VUxfnIqTzDNdOPkU1feiZv1Qau1f/view?usp=sharing

2D 256x256 Dataset:
https://drive.google.com/drive/folders/1YyoQjH1dlb3aOVXdFJ3UI226gsIorvmC?usp=sharing

#### 2. Open and run demo2d.ipynb

# Expected Output

After you run demo2d.ipynb, you should see the testing accuracy listed, the example true vs predicted images, and then the confusion matrix + metrics. There are no outputs to the results folder.

# Pre-trained Model Link:

###### Google Drive Links

2D ResNet-50 Model w/ 76.89% Accuracy:
https://drive.google.com/file/d/1CUZ2VUxfnIqTzDNdOPkU1feiZv1Qau1f/view?usp=sharing

# Acknowledgements:
Thank you to MICrONS and the Allen Institute. 


## About the src Folder

### You will find several notebooks:

#### TeamProjectResnetModelTrain.ipynb
This notebook trains and saves a ResNet model. You can see example training and validation accuracy of the trained
model.

#### TeamProjectExtractImages.ipynb
This notebook utilizes MICrONS tools to extract a dataset of 2D images of excitatory and inhibitory matrix.
The train, val, and test set all have their unique neurons and synapses to avoid data leakage.

#### TeamProjectCNNModelTrain.ipynb
This notebook trains and saves a custom CNN model. You can see example training and validation accuracy of the trained
model.

# 3-D Notebooks

Add information here
