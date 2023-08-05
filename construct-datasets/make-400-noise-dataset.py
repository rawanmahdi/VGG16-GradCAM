#%%
import os
import shutil
import random

datadir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/400-3-cat-breeds-noise/imgs'
destdir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/400-3-cat-breeds-noise'

folders = ['/Persian','/Siamese', '/Tabby']
imgs = []

for img in os.listdir(datadir):
    imgs.append(img)
random.shuffle(imgs)
iterator = iter(imgs)

for folder in folders:
    count = 0
    while(count<=500):
        img =  next(iterator)
        if count<=400:
            shutil.move(datadir+'/'+img, destdir+'/TRAIN'+folder+'/'+img )
            count+=1
        else:
            shutil.move(datadir+'/'+img, destdir+'/TEST'+folder+'/'+img )
            count+=1

# %%
