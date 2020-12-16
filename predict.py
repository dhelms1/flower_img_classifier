import argparse
from functions import *
import json

parser = argparse.ArgumentParser()

parser.add_argument('input', help = 'pass path to image to predict')
parser.add_argument('checkpoint', default='checkpoint.pth', help = 'directory to saved models checkpoint')
parser.add_argument('--arch', default='densenet121', help = 'model architecture to be loaded (same as training)')
parser.add_argument('--top_k', type=int, default=5, help = 'number of top predicted classes to be displayed')
parser.add_argument('--category_names', default='cat_to_name.json', help = 'category to names mapping file')
parser.add_argument('--gpu', default='False', help = 'enable gpu to be used for training', action='store_true')

args = parser.parse_args()

input_img = args.input
checkpoint_path = args.checkpoint
arch_type = args.arch
top_k = args.top_k
cat_names = args.category_names
gpu = args.gpu

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Load in a saved model
loaded_model = load_checkpoint(checkpoint_path, arch_type, gpu)

top_p, top_classes = predict(loaded_model, input_img, gpu, top_k)

class_names = [cat_to_name[c] for c in top_classes]

for prob, name in zip(top_p, class_names):
    print(f'{name}: {round(prob*100,2)}%')
