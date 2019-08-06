''' Implements some useful functions to run predict.py and train.py 
    Last update: 03/30, 06:09
'''

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import torch
import numpy as np
from collections import OrderedDict
from network import Network
from torch import optim

def load_data(data_dir = 'flowers'):
    '''
    Argument:
    --------
    data_dir: string, directory where can find data. Should contain 
    train, valid and test sub-folders
    
    Returns: trainloader, validloader, testloader
    -------
    the dataloaders for the train, valid and test dataset, normalized properly
    '''
    # Define the folders where will fetch the data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms to normalize data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = valid_transforms
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    return trainloader, validloader, testloader, train_data

def load_checkpoint(filepath = 'checkpoint.pth', device = 'cuda'):
    ''' Loads the model as described in file at filepath
        
        Arguments
        ---------
        filepath: path to the file that describes the model and allows us to rebuild it
        device: the device where model should be loaded
        
        Returns
        ---------
        the model as described
    '''
    checkpoint = torch.load(filepath) # load our checkpoint
    # recreate the architecture
    classifier = Network(checkpoint['input_size'], 
                         checkpoint['output_size'], 
                         checkpoint['hidden_layers'], 
                         checkpoint['drop_p'])
    
    # add the parameters
    classifier.load_state_dict(checkpoint['state_dict']) 
    
    # add the attribute that allows us to map classes to indices
    classifier.class_to_idx = checkpoint['class_to_idx']
    
    # load desired pretrained model
    pretrained = checkpoint['pretrained_used']
    model = eval('models.' + pretrained + '(pretrained = True)')
    
    model.classifier = classifier
    
    model.to(device) # move model to device used
    
    return model

def check_on_dataset(model, testloader, criterion, device = 'cuda'):
    ''' Measures accuracy of model's predictions on given data 
    
    Arguments
    ---------
    testloader : dataset to apply model on, containing labels and images
    /!\ data must be correctly formatted w/ appropriate transforms
    criterion: the loss function used
    device: device to be used
    
    Returns: tuple (loss, accuracy)
    ---------
    loss: flat, loss of model on given dataset
    accuracy: float, % of right predictions on given dataset
    '''
    model.eval() # put the model on eval mode
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            predicted = torch.max(output.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / len(testloader), (100 * correct) / total
    
def do_deep_learning(model, trainloader, epochs, criterion, optimizer, validloader, 
                         device = 'cuda', print_every = 50):
    ''' Defines a method to train a model
        Arguments
        ---------
        trainloader: data loader containing train data
        epochs: int, number of epochs
        criterion: loss function used
        optimizer: optimizer used
        validloader: the validation dataset
        device: device to use to run computations
        print_every: int, frequency to get stats on model performance
        
        Side effects
        ---------
        Alongside training, prints, at a (print-every) frequency:
        - the test accuracy: network's accuracy, measured on the test data
        - the validation loss
        - the validation accuracy
    '''
    steps = 0
    model.to(device)
    running_loss = 0
    # just to make sure the model is on training mode
    model.train()
    # for each epoch...
    for e in range(epochs):
        # for each batch of images in the train data...
        for inputs, labels in trainloader:
            print('Step: {}.. '.format(steps))
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Track loss and accuracy on validation test 
            if steps % print_every == 0:
                model.eval() # put the model on eval mode
                # turn off the gradients to save memory and computation
                with torch.no_grad():
                    valid_loss, valid_accuracy = check_on_dataset(model, validloader, 
                                                                     criterion, device)
                    print("Epochs: {}/{}.. ".format(e + 1, epochs),
                          "Training loss: {:.3f}.. ".format(running_loss / print_every), 
                          "Valid loss: {:.3f}.. ".format(valid_loss),
                          "Valid accuracy: {:.3f} %.. ".format(valid_accuracy))
                    running_loss = 0
                    model.train() # put model back on training

def check_on_testdata(model, testloader, criterion, device):
    '''when given a testloader for test data, returns check stats
    on test set
    '''
    model.eval() # put the model on eval mode
    with torch.no_grad():
        test_loss, test_accuracy = check_on_dataset(model, testloader, 
                                                    criterion, device)
        print("Test accuracy: {:.3f} %..".format(test_accuracy), 
              "Test loss: {:.3f}".format(test_loss))
        
def save_checkpoint(model, input_size, output_size, hidden_layers, 
                        drop_p, train_data, epochs, optimizer, 
                        pretrained_used = 'vgg16',
                        filename = 'checkpoint.pth'):
    ''' Saves a given neural network model
    Arguments
    ---------
    model: Pytorch neural network, model to save
    input_size: int, the size of the input
    output_size: int, the size of the output
    hidden_layers: list of ints, the sizes of each of the hidden layers
    drop_p: float, the probability of dropout
    train_data: a dataset of images, used to save the mapping from idx to clsses
    epochs: int, the nb of epochs used to do the training
    optimizer: optimizer used to train
    pretrained_used: name of the pretrained model used for features dedection
    filename: string, name of the file where we are going to save the network.
    Default: 'checkpoint.pth'
    '''
    # Define a dictionary w/ all information necessary to rebuild the model
    checkpoint = {'input_size':input_size, 
                  'output_size':output_size, 
                  'hidden_layers':hidden_layers,
                  'drop_p':drop_p,
                  'state_dict':model.state_dict(), 
                  'epochs':epochs,
                  'optimizer_state_dict':optimizer.state_dict(),
                  'class_to_idx':train_data.class_to_idx,
                  'pretrained_used':pretrained_used
                 }
    # Save the dict 
    torch.save(checkpoint, filename)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    side_resize = 256
    img = image.resize((side_resize,side_resize))
    
    # crop out the center 224x224 portion of the image
    side_crop = 224
    width, height = img.size
    left = (width - side_crop) / 2
    right = left + side_crop
    upper = (height - side_crop) / 2
    lower = upper + side_crop
    img = img.crop((left, upper, right, lower))
    
    np_image = np.array(img) # convert PIL image to np-array
    
    # normalize data w/ given model's normalization
    np_image = np_image / 255 # squish
    means = np.array([0.485, 0.456, 0.406])
    stdvs = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stdvs
    
    # PyTorch expects the color channel to be the first dimension
    return np.transpose(np_image, (2,1,0))

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, device, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Arguments
        ---------
        image_path: string, path to the image file to process
        model: pytorch neural network, (trained) model used for inference
        device: desired device
        topk: int, number of most likely classes to return
        
        Returns: 
        ---------
        probs: list of floats, the topk most likely class, as predicted by the model
        classes: list of strings, the most lilely classes attached to these probabilities
        
        /!\ Since the model returns indices, we want to do 
        idx -> classes -> class names, to return actual class names
    
    '''
    
    model.to(device) # move the model to device
    
    model.eval() # put model on eval mode
    
    # load image and transform to np array
    with Image.open(image_path) as img:
        np_img = process_image(img)
        
    # transform np array to FloatTensor
    tensor_img = torch.from_numpy(np_img).type(torch.FloatTensor)
    tensor_img = tensor_img.to(device) # move to device
    
    # reshape the float tensor
    tensor_img.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(tensor_img) # get the logits
        
    ps = torch.exp(output) # transform the logits into probabilities
    
    # get the top  𝐾  largest values
    probs, indices = ps.topk(topk)
    probs, indices = probs.to("cpu").numpy(), indices.to("cpu").numpy()
    
    # transform  indices -> class numbers
    #### print('class to idx {}.. '.format(model.classifier.class_to_idx))
    idx_to_class = {int(value):str(key) for key,value in model.classifier.class_to_idx.items()}
    #### print("idx to class {}.. ".format(idx_to_class))
    #### print('shape of indices : {}.. '.format(indices.shape))
    #### print('shape of probs[0] : {}.. '.format(probs[0].shape))
    classes = [idx_to_class[val] for val in indices[0]]
    #### print('probs: {}'.format(probs), 'indices: {}'.format(indices))
    print("predicted classes: {}.. ".format(classes))
    
    # return two lists with the probabilities and the class numbers
    return probs[0].tolist(), classes

def class_to_name(cat_number, cat_to_name):
    ''' Given class number and cat_to_name dictionary, returns class names
        Arguments
        ---------
        cat_number: list of values (in this context, strings), describing class numbers
        cat_to_name: dictionary that gives the cat_number -> cat_name correspondance
        
        Returns: 
        ---------
        class: list of string that is the corresponding class
    '''
    return [cat_to_name[cat] for cat in cat_number]