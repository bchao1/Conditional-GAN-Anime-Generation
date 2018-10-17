# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:49:51 2018

@author: USER
"""

import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

import datasets
import CGAN
import utils

if __name__ == '__main__':
    
    ########## Configuring stuff ##########
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))
    
    latent_dim = 100
    hair_classes = 12
    eye_classes = 10
    num_classes = hair_classes + eye_classes
    
    gan_type = 'CGAN'
    batch_size = 64
    steps = 100000
    smooth = 0.9
    config = gan_type
    print('Configuration: {}'.format(config))
    
    
    root_dir = '../data/images'
    tags = '../data/tags.pickle'
    sample_dir = './samples/{}'.format(config)
    ckpt_dir = './checkpoints/{}'.format(config)
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    ########## Start Training ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])
    train_data = datasets.Anime(root_dir = root_dir,
                                tags = tags,
                                 transform = transform)
    shuffler = datasets.Shuffler(dataset = train_data, 
                                 batch_size = batch_size)
    
    
    G = CGAN.Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = CGAN.Discriminator(num_classes = num_classes).to(device)
    
    G_optim = optim.Adam(G.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    D_optim = optim.Adam(D.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    
    discrim_log = []
    gen_log = []
    criterion = torch.nn.BCELoss()
    
    for step_i in range(1, steps + 1):
        # Create real and fake labels (0/1)
        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
        
        ########## Training the Discriminator  ################################

        real_img, hair_class, eye_class = shuffler.get_batch()
        real_img = real_img.to(device)
        hair_class, eye_class = hair_class.to(device), eye_class.to(device)
        correct_class = torch.cat((hair_class, eye_class), 1)
        wrong_hair = utils.mismatch(hair_class).to(device)
        wrong_eye = utils.mismatch(eye_class).to(device)
        wrong_class = torch.cat((wrong_hair, wrong_eye), 1)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_img = G(z, correct_class).to(device)
            
        real_img_correct_class = D(real_img, correct_class)
        real_img_wrong_class = D(real_img, wrong_class)
        fake_img_correct_class = D(fake_img, correct_class)
                        
        discrim_loss = (criterion(real_img_correct_class, real_label) +
                        (criterion(real_img_wrong_class, fake_label) +
                        criterion(fake_img_correct_class, fake_label)) * 0.5)
        D_optim.zero_grad()
        discrim_loss.backward()
        D_optim.step()
            
        ########## Training the Generator ##########
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_img = G(z, correct_class).to(device)
        fake_img_real_class = D(fake_img, correct_class)
        gen_loss = criterion(fake_img_real_class, real_label)
                
        G_optim.zero_grad()
        gen_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        discrim_log.append(discrim_loss.item())
        gen_log.append(gen_loss.item())
        utils.show_process(steps, step_i, gen_log, discrim_log)
        ########## Checkpointing ##########
        
        if step_i == 1:
            save_image(utils.denorm(real_img), 
                           os.path.join(sample_dir, 'real.png'))
        if step_i % 500 == 0:
            save_image(utils.denorm(fake_img), 
                           os.path.join(sample_dir, 'fake_step_{}.png'.format(step_i)))
        if step_i % 2000 == 0:
            utils.save_model(G, G_optim, step_i, tuple(gen_log), 
                                     os.path.join(ckpt_dir, 'G_{}.ckpt'.format(step_i)))
            utils.save_model(D, D_optim, step_i, tuple(discrim_log), 
                                     os.path.join(ckpt_dir, 'D_{}.ckpt'.format(step_i)))
            #utils.plot_loss(gen_log, discrim_log, os.path.join(ckpt_dir, 'loss.png'))
            utils.generation_by_attributes(G, device, latent_dim, hair_classes, eye_classes, step_i, sample_dir)

        

    
    
    