import os
import shutil

datadir = 'C:/Users/Rawan Alamily/Downloads/huge-cat-data'
destdir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/2500-cat-breed'

for folder in ['/Bengal', '/Persian', '/Ragdoll', '/Siamese', '/Tabby']:
    count = 0
    for img in os.listdir(datadir+folder):
        if count<2800 :
            shutil.move(datadir+folder+img, destdir+'/new'+folder+img)
            count+=1
        else:
            break
