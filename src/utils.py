# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:47:07 2018

@author: USER
"""

import torch
import torch.nn
import os


import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

hair_mapping =  ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 
                 'pink', 'blue', 'black', 'brown', 'blonde']

eye_mapping = ['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 
               'brown', 'red', 'blue']
               
def denorm(img):
    output = img / 2 + 0.5
    return output.clamp(0, 1)

def save_model(model, optimizer, step, log, file_path):
    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step,
             'log' : log}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_epoch = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_epoch, log

def show_process(steps, step_i, gen_log, discrim_log):
    print('Step {}/{}: G_loss [{:8f}], D_loss [{:8f}]'.format(
            step_i,
            steps,
            gen_log[-1], 
            discrim_log[-1]))
    return

def plot_loss(gen_log, discrim_log, file_path):
    epochs = list(range(len(gen_log)))
    plt.semilogy(epochs, gen_log)
    plt.semilogy(epochs, discrim_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} epochs)".format(len(epochs)))
    plt.savefig(file_path)
    plt.close()
    return

def mismatch(target):
    batch_size = target.shape[0]
    classes = target.shape[1]
    wrong = torch.zeros(target.shape)
    for i in range(batch_size):
        c = torch.max(target[i, :], 0)[1]
        shifted = (c + np.random.randint(classes - 1)) % classes
        wrong[i][shifted] = 1
    return wrong

def generation_by_attributes(model, device, latent_dim, hair_classes, eye_classes, step, sample_dir):
    
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    hair_class = np.random.randint(hair_classes)
    eye_class = np.random.randint(eye_classes)

    for i in range(64):
    	hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1

    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, tag)
    file_path = '{} hair {} eyes, step {}.png'.format(hair_mapping[hair_class], eye_mapping[eye_class], step)
    save_image(denorm(output), os.path.join(sample_dir, file_path))
    