# Conditional GAN Anime Generation
Conditional Anime Generation using conditional GAN.

## Model and Objectives
![model](./img_src/conditional_gan.PNG)

Different from vanilla GAN, the generator and discriminator in conditional GAN (abbr. CGAN) will be given a specific condition (could be a one-hot vector indicating classes or a word-embedding).
***
The discriminator considers the following cases:  
1. Real distribution, correct conditon -> positive
2. Real distribution, wrong condition -> negative
3. Fake distribution, real condition -> negative
***
Our training process roughly follows the paper: https://arxiv.org/abs/1605.05396, only that the text descriptions are replaced with a one-hot class vector.

![cgan algo](./img_src/cgan_algo.png)

## Results
|Fixing noise|
|------------|
|![fix noise](./results/fix_noise_1.png)|

|Changing eye color|
|------------------|
|![change eye](./results/change_eye_color_1.png)|
|![change eye](./results/change_eye_color_2.png)|
|![change eye](./results/change_eye_color_3.png)|

|Change hair color|
|-----------------|
|![change hair](./results/change_hair_color_1.png)|
|![change hair](./results/change_hair_color_2.png)|
|![change hair](./results/change_hair_color_3.png)|

|Condition|Generated|
|---------|---------|
|blonde hair, purple eyes|![gen](./results/blonde_hair_purple_eyes.png)|
|blue hair, red eyes|![gen](./results/blue_hair_red_eyes.png)|
|white hair, green eyes|![gen](./results/white_hair_green_eyes.png)|

