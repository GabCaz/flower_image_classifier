class predict:
    ''''
    last update: 05:47
    
    pass in a single image /path/to/image and return the predicted flower name and class probability
    using a trained network
    
    Summary of arguments:
    ----------------------
    /path/to/image: filepath to image to do prediction on
    checkpoint: model checkpoint to use for prediction
    --top_k desired_int: specify int, number of most likely classes to return
    --category_names cat_to_name.json: giving correspondance class -> class name
    --gpu: choose to use gpu for inference or not
    
    Basic usage:
    -----------
    python predict.py /path/to/image checkpoint
    
    Options:
    -------
        Return top KK most likely classes: python predict.py input checkpoint --top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        Use GPU for inference: python predict.py input checkpoint --gpu
    '''
    import argparse # argparse to take command-like arguments
    import json
    import torch
    from torch import nn
    import torch.nn.functional as F
    import utility_functions as uf
    from network import Network
    from PIL import Image
    import numpy as np
    
    # get in-line arguments
    parser = argparse.ArgumentParser() # creates an argument parser object
    parser.add_argument('filepath', nargs = '?', action = 'store', 
                        default = 'flowers/test/16/image_06657.jpg', type = str)
    parser.add_argument('checkpoint', nargs = '?', action = 'store',
                        default = 'checkpoint.pth', type = str)
    parser.add_argument('--top_k', dest = 'topk', nargs = '?', action = 'store', 
                        default = 1, type = int)
    parser.add_argument('--category_names', dest = 'cat_names', nargs = '?', 
                        action = 'store', default = 'cat_to_name.json')
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', default = False) # use GPU or not?
    
    args = parser.parse_args()
    
    # get desired device
    device = torch.device("cuda" if args.gpu else "cpu") 
    
    # load model in
    model = uf.load_checkpoint(args.checkpoint, device)
    
    # make sure we got a fine model loaded in
    ### trainloader, validloader, testloader, train_data = uf.load_data('flowers')
    ### uf.check_on_testdata(model, testloader, nn.NLLLoss(), device)
        
    # get predictions
    topk = args.topk
    probs, classes = uf.predict(args.filepath, model, device, topk)
    
    # get correspondance 
    with open(args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # get class names
    class_names = uf.class_to_name(classes, cat_to_name)
    
    # print the topk most likely classes w/ the corresponding probabilities
    # (topk default to 1, so it respects desired behavior if unspecified)
    print("the {} highest probabilities are: {}".format(topk, probs),
          "the corresponding classes: {}".format(class_names))