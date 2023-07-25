#%%
import os
import shutil

datadir = 'C:/Users/Rawan Alamily/Downloads/huge-cat-data/images'
destdir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/400-cat-breeds'

for folder in ['/Bengal', '/Persian', '/Ragdoll', '/Siamese', '/Tabby']:
    count = 0
    for img in os.listdir(datadir+folder):
        if count<=400:
            shutil.copy(datadir+folder+'/'+img, destdir+'/TRAIN'+folder+'/'+img )
            count+=1
        elif count<=500:
            shutil.copy(datadir+folder+'/'+img, destdir+'/TEST'+folder+'/'+img )
            count+=1
        else:
            break

# %%
