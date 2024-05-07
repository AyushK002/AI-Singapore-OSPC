# Standard Python Library
import sys
import logging
import random
import csv
# Import opencv
import cv2
import argparse

# Import pytesseract
import pytesseract
from pytesseract import Output

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import save_img
import tensorflow_text as text
import shutil
import numpy as np
import os
import json
from bs4 import BeautifulSoup
import re
from functools import partial

# Import main
from main import main
from utils import str2bool

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lightning').setLevel(0)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

def get_arg_parser_m():
    parser = argparse.ArgumentParser(description='Training and evaluation script for hateful memes classification')

    parser.add_argument('--dataset', default='mcc_run') #, choices=['hmc','harmeme','Singapore_defense','mcc_run'])
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--num_mapping_layers', default=1, type=int)
    parser.add_argument('--map_dim', default=768, type=int)

    parser.add_argument('--fusion', default='align',
                        choices=['align', 'concat'])

    parser.add_argument('--num_pre_output_layers', default=3, type=int)

    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.2, 0.4, 0.1],
                        help="Set drop probabilities for map, fusion, pre_output")

    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--log_every_n_steps', type=int, default=25)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    parser.add_argument('--proj_map', default=False, type=str2bool)

    parser.add_argument('--pretrained_proj_weights', default=True, type=str2bool)
    parser.add_argument('--freeze_proj_layers', default=True, type=str2bool)

    parser.add_argument('--comb_proj', default=True, type=str2bool)
    parser.add_argument('--comb_fusion', default='align',
                        choices=['concat', 'align'])
    parser.add_argument('--convex_tensor', default=False, type=str2bool)

    parser.add_argument('--text_inv_proj', default=True, type=str2bool)
    parser.add_argument('--phi_inv_proj', default=True, type=str2bool)
    parser.add_argument('--post_inv_proj', default=True, type=str2bool)

    parser.add_argument('--enh_text', default=True, type=str2bool)

    parser.add_argument('--phi_freeze', default=True, type=str2bool)

    parser.add_argument('--name', type=str, default='text-inv-comb',
                        choices=['adaptation', 'hate-clipper', 'image-only', 'text-only', 'sum', 'combiner', 'text-inv',
                                 'text-inv-fusion', 'text-inv-comb']
                        )
    parser.add_argument('--pretrained_model', type=str, default='text-inv-comb_Singapore_defense-frozen_comb.ckpt') #'harmeme_text-inv-comb_best.ckpt')
    parser.add_argument('--reproduce', default=True, type=str2bool)
    parser.add_argument('--print_model', default=False, type=str2bool)
    parser.add_argument('--fast_process', default=False, type=str2bool)

    return parser

#import cv2

#from PIL import Image
#import numpy as np


def preprocess_final(im):
    if(len(im.shape)==3 and im.shape[2]>3):
        im=im[:,:,:3]
    im= cv2.bilateralFilter(im,5,55,60)
    if len(im) == 3:
      im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 240, 255, 1)
    return im

# im = np.array(Image.open('img4.png'))
# text = pytesseract.image_to_string(im)
# print(text.replace('\n', ' '))

custom_config = r"--oem 3 --psm 6 -l eng+chi_sim"

#img=np.array(Image.open('img/8b52mg.jpg'))
#im=preprocess_final(img)
#text = pytesseract.image_to_string(im, config=custom_config)
#print(text.replace('\n', ''))
def is_jpg(filename):
    # Check if the file extension is '.jpg' or '.jpeg'
    return os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg']


if __name__ == "__main__":
    # Iteration loop to get new image filepath from sys.stdin:
    with open('run.csv', 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        writer.writerow(['id','image','label','text','split'])
        ind = 0
        for line in sys.stdin:
            # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
            image_path = line.rstrip()

            # INSERT GET_MEME_TEXT FUNCTION HERE
            # 1. Open image filepath ========================================= #
            #im = cv2.imread(image_path)
            # 2. Get meme text =============================================== #
            #textread, coordinates = preprocess_final(image=im)
            
            if is_jpg(image_path):
                img=np.array(Image.open(image_path))
                im=preprocess_final(img)

            else:
    # Convert RGBA image to RGB before saving as JPEG
                img=Image.open(image_path)
                image = img.convert('RGB')
                image.save("image.jpg", "JPEG", quality=100)
                img=np.array(Image.open('image.jpg'))
                im=preprocess_final(img)
            
                os.remove("image.jpg")
             #(f'/content/drive/MyDrive/ISSUES/resources/datasets/Singapore_defense/TD_Memes/img_41{i}.jpg'))
            #im=preprocess_final(img)
            textread = pytesseract.image_to_string(im, config=custom_config)
            textread.replace('\n','')
                               
            item = [ind, image_path, 1, textread, 'test']
            ind = ind+1
            writer.writerow(item)



    try:
        # Process the image
        #proba, label = process_line_by_line(filepath=image_path)

        # Ensure each result for each image_path is a new line
        #sys.stdout.write(f"{proba:.4f}\t{label}\n")

        pars = get_arg_parser_m()
        args = pars.parse_args([])
        # Write the name of the dataset to specify 
        main(args)


        # Load the model
        '''
        model = HateClassifier.load_from_checkpoint(f'resources/pretrained_models/{args.pretrained_model}',
                                                    args=args)

        # Create a test data loader
        test_dataset = load_dataset(args=args, split='test')
        test_data_loader = DataLoader(test_dataset, batch_size=ind)

        # Iterate over the test data loader and make predictions
        predictions = []
        for batch in test_data_loader:
            predictions.append(model(batch))

        # Calculate the accuracy of the model
        #accuracy = torch.mean(torch.eq(predictions, test_labels))

        sys.stdout.write(predictions)
        '''
    except Exception as e:
        # Output to any raised/caught error/exceptions to stderr
        sys.stderr.write('') #str(e))

    
    












