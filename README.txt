Author: Gabriel Cazaubieilh,  instructions given by Udacity, within Udacity's Python for AI Nanodegree
Last changed: August 4, 2019

This program implements a Neural Network, using transfer learning, to classify flower breeds.

The repository includes the following files:
    - The Image Classifier Project.html is an overview of the project, with guidelines given by Udacity. It is a converted Jupyter Notebook.

    - utility_functions Implements some useful functions to run predict.py and train.py, notably:
        . to load data (defining transforms and data loaders for testing, validation and training sets)
        . recreate a network from a file 'checkpoint.pth' that contains its information
        . check_on_dataset, which computes model accuracy and loss on a given data set
        . do_deep_learning, which is used to train the model
        . save_checkpoint, to save the model information in a file
        . process_image: scales, crops, and normalizes a PIL image for a PyTorch model, eturns an Numpy array
        . predict(image_path, model, device, topk=1), to make predictions using the model

    - train.py can be executed to train a new network on a dataset and save the model as a checkpoint
    Prints out training loss, validation loss, and validation accuracy as the network trains

    - predict.py can be executed to pass in a single image /path/to/image and return the predicted flower name and class probability using a trained network

    - network.py builds the constructor we will use to construct our model, using ReLU activations and dropout
