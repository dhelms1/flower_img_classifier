import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
import warnings


def data_loader(data_dir):
    '''
    data_dir -- a directory with train and validation folders within
    '''
    train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    
    return trainloader, train_data, validloader, valid_data


def create_model(arch_type, hidden_units):
    '''
    Create a new model based on the architecture and hidden units specified
    '''
    warnings.simplefilter("ignore", UserWarning)
    model = getattr(models, arch_type)(pretrained=True)
    input_size = model.classifier.in_features # get original classifier input size
    
    for param in model.parameters():
        param.require_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    return model

def train_model(model, trainloader, validloader, epochs, gpu_flag, criterion, optimizer):
    '''
    Train a model for a given number of epochs.
    Move model to either cpu or gpu depending on parameters.
    '''
    device = torch.device('cuda' if (gpu_flag==True & torch.cuda.is_available()) else 'cpu')
    model.to(device)
    
    print(f'Training {model.__class__.__name__} with {device}')
    
    train_loss = 0
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    
                    log_ps = model.forward(images)
                    valid_loss += criterion(log_ps, labels).item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(k=1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
        model.train()
        
        print(f"Epoch {epoch+1}/{epochs}... "
              f"Train Loss: {train_loss/len(trainloader):.3f}... "
              f"Valid Loss: {valid_loss/len(validloader):.3f}... "
              f"Valid Accuracy: {accuracy/len(validloader):.3f}")
        train_loss = 0 
        model.train() 
    
    return model, optimizer

def save_model(model, optimizer, train_data, save_dir):
    '''
    Save a checkpoint for a trained model
    '''
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'optim_state': optimizer.state_dict,
                  'classifier': model.classifier}
    
    torch.save(checkpoint, save_dir)
    return None

def load_checkpoint(checkpoint_path, arch_type, gpu_flag):
    '''
    Load a model from a saved checkpoint
    '''
    warnings.simplefilter("ignore", UserWarning)
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, arch_type)(pretrained=True)
    for param in model.parameters():
        param.require_grad = False
    device = torch.device('cuda' if (gpu_flag==True & torch.cuda.is_available()) else 'cpu')
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    '''
    img = Image.open(image) # open image
    img.thumbnail((256,256), Image.ANTIALIAS) # resize to smallest side 256
    
    # Center crop
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    np_image = np.array(img)
    
    np_image = np_image/255 # convert to [0,1]
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # normalize image to same as model training
    img_final = np_image.transpose(2,0,1) # change channels to match PyTorch
    
    return img_final

def predict(model, image_path, gpu_flag, topk):
    '''
    Predict the class (or classes) of an image using a trained deep learning model
    '''
    img = process_image(image_path)
    image = torch.from_numpy(np.expand_dims(img, axis=0)).type(torch.FloatTensor)
    device = torch.device('cuda' if (gpu_flag==True & torch.cuda.is_available()) else 'cpu')
    model.eval() # turn off dropout
    with torch.no_grad():
        image = image.to(device)
        print(f'Predicting with {device}')
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)   
        model.train()
    
    idx_to_class = {i:c for c,i in model.class_to_idx.items()} # invert dictionary
    top_p = top_p.reshape(-1).tolist() # convert 2D tensor to list
    top_class = top_class.reshape(-1).tolist() # convert 2D tensor to list
    top_class = [idx_to_class[tc] for tc in top_class] # convert from idx to label
    
    return top_p, top_class