# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:40:51 2021

@author: Jerry
"""
from args import init_arguments
import numpy as np
import array
import struct
import matplotlib.pyplot as plt
import cv2
import math
import os

# In[]

def read_image(path):  #path = os.path.join('data/', images[0])
    file_extention = path.split('.')[-1]
    
    if file_extention == 'bmp':
        f = open(path, 'rb')
        bmp_header_b = f.read(0x36) # read top 54 byte of bmp file 
        bmp_header_s = struct.unpack('<2sI2H4I2H6I', bmp_header_b) # parse data with struct.unpack()
        pixel_index = bmp_header_s[4] - 54
        bmp_rgb_data_b = f.read()[pixel_index:] # read pixels of bmp file 
        list_b = array.array('B', bmp_rgb_data_b).tolist()
        rgb_data_3d_list  = np.reshape(list_b, (bmp_header_s[6], bmp_header_s[7], bmp_header_s[8])).tolist() # reshape pixel with height, width, RGB channel of image
        
        image = []
        for row in range(len(rgb_data_3d_list)):
            image.insert(0, rgb_data_3d_list[row])
        
        
    elif file_extention == 'raw':
        f = open(path, 'rb').read()
        image = reshape_byte(f) # reshape byte
        
    image = np.array(image) # store into np.array
    
    if len(image.shape) != 3:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    
    return image

def reshape_byte(byte, size=[512,512]): 
    new_img = []
    
    for row in range(size[0]):
        new_img_row = []
        
        for col in range(size[1]):
            new_img_row.append(byte[row * size[1] + col])
            
        new_img.append(new_img_row)
    
    # new_img = np.reshape(new_img, (size[0], size[1], 1))
    
    return new_img

def negativate(image): 
    
    new_img = 255 - image
    
    return new_img


def logTransform(c, f): # Compute log

    g = c * math.log(float(1 + f), 10);
    
    return g

def logTransform_image(image, outputMax=255, inputMax=255, logarithm = 10): # Apply logarithmic transformation for an image  
    
    # method 1    
    # c = outputMax / np.log([inputMax + 1])
    # image = c * (np.log(image + 1))
    
    # method 2
    # Read pixels and apply logarithmic transformation 
    c = outputMax / math.log(inputMax + 1, logarithm)
    for i in range(0, image.shape[0]):

        for j in range(0, image.shape[1]):
            
            ## Get pixel value at (x,y) position of the image
            f = image[i][j] 
            
            ## Do log transformation of the pixel
            f = round(logTransform(c, f)) 

            ## Modify the image with the transformed pixel values
            image[i][j] = f 

    return image

def gamma_Transform(image, gamma = 2):
    
    # new_image = image^gamma
    new_image = np.power(image / float(np.max(image)), gamma)
    
    return new_image

def get_image_center(image, center_size = 10): # image = original
    height, width, channel = image.shape
    center_x = height // 2
    center_y = width // 2
    center_half_size = center_size // 2
    center_image = image[center_x - center_half_size : center_x + center_half_size,
                         center_x - center_half_size : center_y + center_half_size]
    
    return center_image


def NNI(image, height_new, width_new): # Nearest Neighbor Interpolation # height_new = 1024 width_new = 512
    height_old, width_old, channel_old = image.shape # get shape of image = original
    new_image = np.zeros((height_new, width_new, channel_old))#, dtype=np.uint8)
    for i in range(height_new): 
        for j in range(width_new):
            scale_x = round((i + 1) * (height_old / height_new))
            scale_y = round((j + 1) * (width_old / width_new))
            new_image[i, j] = image[scale_x - 1, scale_y - 1]
    return new_image

def BiLinear(image, height_new, width_new): # BiLinear Interpolation 
    height_old, width_old, channel_old = image.shape # get shape of image = original.copy()
    image = np.pad(image, ((0, 1), (0, 1), (0, 0)), 'constant') #padding
    new_image = np.zeros((height_new, width_new, channel_old))#, dtype=np.uint8)
    for i in range(height_new):
        for j in range(width_new):
            scale_x = (i + 1) * (height_old / height_new) - 1 # get x scale from original size and target size
            scale_y = (j + 1) * (width_old / width_new) - 1 # get y scale from original size and target size
            x = math.floor(scale_x) # get integer of x scale
            y = math.floor(scale_y) # get integer of y scale
            u = scale_x - x # get decimal of x scale
            v = scale_y - y # get decimal of y scale
            
            # calculate new value of pixel of target size by fomula
            new_image[i, j] = image[x, y] * (1 - u) * (1 - v) + \
                            image[x + 1, y] * u * (1 - v) + \
                            image[x, y + 1] * (1 - u) * v  + \
                            image[x + 1, y + 1] * u * v
            
    return new_image

def up_and_downsampling(image, image_name, save_path, method_name = 'BiLinear'):
    
    if method_name == 'BiLinear':
        sample_method = BiLinear
    elif method_name == 'NNI':
        sample_method = NNI
    else:
        print('Please choose "BiLinear" or "NNI"')
    
    # original = image
    plt.figure(figsize = (24, 18))
    parameters = {'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'axes.titlesize': 40}
    
    plt.rcParams.update(parameters)
    
    plt.subplot(2,3,1)
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    plt.title('Original')
    
    
    image_512to128 = sample_method(image.copy(), 128, 128)
    plt.subplot(2,3,2)
    plt.imshow(image_512to128, cmap='gray')
    # plt.axis('off')
    plt.title(f'{method_name} - 512x512 to 128x128')
    
    image_512to32 = sample_method(image.copy(), 32, 32)
    plt.subplot(2,3,3)
    plt.imshow(image_512to32, cmap='gray')
    # plt.axis('off')
    plt.title(f'{method_name} - 512x512 to 32x32')
    
    image_32to512 = sample_method(image_512to32.copy(), 512, 512)
    plt.subplot(2,3,4)
    plt.imshow(image_32to512, cmap='gray')
    # plt.axis('off')
    plt.title(f'{method_name} - 32x32 to 512x512')
    
    image_512to1024 = sample_method(image.copy(), 1024, 512)
    plt.subplot(2,3,5)
    plt.imshow(image_512to1024, cmap='gray')
    # plt.axis('off')
    plt.title(f'{method_name} - 512x512 to 1024x512')
    
    image_128to256 = sample_method(image_512to128.copy(), 256, 512)
    plt.subplot(2,3,6)
    plt.imshow(image_128to256, cmap='gray')
    # plt.axis('off')
    plt.title(f'{method_name} - 128x128 to 256x512')
    
    plt.savefig(os.path.join(save_path, f'up and downsampling of {image_name}-{method_name}.png'))
    plt.close()
    
    
def main(args):
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    images = os.listdir(args.image_path)
    
    for i_image_name in images:
        # print(i_image_name)
        
        image_path = os.path.join(args.image_path, i_image_name)
        
        original_image = read_image(image_path)
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, f'original {i_image_name}.png'))
        plt.close()
        
        # plot center of image
        center_image = get_image_center(original_image.copy())
        plt.imshow(center_image, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, f'center of {i_image_name}.png'))
        plt.close()
        
        # negative
        negative = negativate(original_image.copy())
        plt.imshow(negative, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, f'negative of {i_image_name}.png'))
        plt.close()
        
        # log-transform
        log_trans = logTransform_image(original_image.copy())
        plt.imshow(log_trans, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, f'log-transform of {i_image_name}.png'))
        plt.close()
        
        plt.figure(figsize = (24, 18))
        parameters = {'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'axes.titlesize': 40}
        
        plt.rcParams.update(parameters)
        
        logarithm_list = [2, 5, 10, 100]
        for index, logarithm in enumerate(logarithm_list):
            
            log_trans = logTransform_image(original_image.copy(), logarithm)
            plt.subplot(2, 2, index + 1)
            plt.imshow(log_trans, cmap='gray')
            plt.title(f'logarithm = {logarithm}')
            
        plt.savefig(os.path.join(args.save_path, f'log-transform of {i_image_name}.png'))
        plt.close()
        
        # gamma-transform
        plt.figure(figsize = (24, 18))
        parameters = {'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'axes.titlesize': 40}
        
        plt.rcParams.update(parameters)
        gamma_list = [0.25, 0.5, 2, 4]
        
        for index, gamma in enumerate(gamma_list):
            
            gamma_trans = gamma_Transform(original_image.copy(), gamma)
            plt.subplot(2, 2, index + 1)
            plt.imshow(gamma_trans, cmap='gray')
            plt.title(f'gamma = {gamma}')
            
        plt.savefig(os.path.join(args.save_path, f'gamma-transform of {i_image_name}.png'))
        plt.close()
        
        # up_and_downsampling
        up_and_downsampling(original_image, i_image_name, args.save_path, args.method_name)
    

# In[]

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)

