import argparse
from functions import *
from torch import optim
from torch import nn

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help = 'pass data directory to read images from')
parser.add_argument('--save_dir', default='checkpoint.pth', help = 'directory to save model checkpoints')
parser.add_argument('--arch', default='densenet121', help = 'model architecture to be used')
parser.add_argument('--learning_rate', type=float, default=0.001, help = 'learning rate of the optimizer')
parser.add_argument('--hidden_units', type=int, default=1000, help = 'number of hidden units to be used in the model')
parser.add_argument('--epochs', type=int, default=5, help = 'number of epochs to train the model for')
parser.add_argument('--gpu', default='False', help = 'enable gpu to be used for training', action='store_true')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch_type = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# Load data from directory
trainloader, train_data, validloader, valid_data = data_loader(data_dir)

# Create the Model
model = create_model(arch_type, hidden_units)

# Set up Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Train the Model
model, optimizer = train_model(model, trainloader, validloader, epochs, gpu, criterion, optimizer)

# Save the Model
save_model(model, optimizer, train_data, save_dir)