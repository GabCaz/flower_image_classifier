class train:
    '''
    Can be run to train a new network on a dataset and save the model as a checkpoint
    Prints out training loss, validation loss, and validation accuracy as the network trains
    
    Last update: 03/30, 05:47
    
    Basic usage:
    -----------
    python train.py data_directory
    
    Summary of Arguments:
    ----------
    data_directory
    --save_dir
    --arch
    --learning_rate
    --hidden_units
    --epochs
    --gpu
    
    Options:
    --------
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units [512] --epochs 20
    Use GPU for training: python train.py data_dir --gpu
    '''
    # IMPORTS
    import argparse # argparse to take command-like arguments
    import utility_functions as uf
    from network import Network
    import torch
    from torch import optim
    from torch import nn
    from torchvision import transforms, datasets, models
    # import numpy as np
    # import torch.nn.functional as F
    
    # get arguments typed in using arparse '''
    parser = argparse.ArgumentParser() # creates an argument parser object
    parser.add_argument('data_dir', nargs = '?', action = 'store', default = 'flowers') # data dir
    parser.add_argument('--save_dir', dest = 'save_dir', nargs = '?', action = 'store', 
                        default = 'checkpoint.pth')
    parser.add_argument('--arch', nargs = '?', dest = 'arch', action = 'store', default = 'vgg16')
    parser.add_argument('--learning_rate', nargs = '?', dest = 'lr', default = 0.001, type = float)
    parser.add_argument('--hidden_units', nargs = '?', dest = 'hidden_layers', default = [1000, 500])
    parser.add_argument('--epochs', nargs = '?', dest = 'epochs', default = 3, type = int)
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', default = False) # use GPU or not?
    args = parser.parse_args()
    #### print(args)
    
    device = torch.device("cuda" if args.gpu else "cpu") # get desired device
    
    # import data from desired folder
    trainloader, validloader, testloader, train_data = uf.load_data(data_dir = args.data_dir)
    
    # load pretrained model of desired shape
    arch = args.arch
    model = eval('models.' + arch + '(pretrained = True)')
    
    # freeze the model's parameters
    for param in model.parameters():
        param.required_grad = False
        
    # Define a new, untrained feed-forward network as a classifier
    # using ReLU activations and dropout
    input_size = 25088
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    output_size = 102
    dropout_p = 0.3
    classifier = Network(input_size, output_size, hidden_layers, dropout_p)
    
    # replace the model's classifier w/ ours
    model.classifier = classifier
    
    criterion = nn.NLLLoss() # since used log-softmax, use NLLLoss as criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    
    # Train the classifier layers using the pre-trained network to get the features
    uf.do_deep_learning(model, trainloader, epochs, criterion, 
                             optimizer, validloader, device)
    
    # Check the network against test set
    uf.check_on_testdata(model, testloader, criterion, device)
    
    # Save our trained classifier in a checkpoint
    uf.save_checkpoint(model.classifier, input_size, output_size, hidden_layers, dropout_p,
                            train_data, epochs, optimizer, arch, args.save_dir)