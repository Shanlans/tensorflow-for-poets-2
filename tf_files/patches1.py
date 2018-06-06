# -*- coding: utf-8 -*-
import numpy as np
from skimage import io,util
import os


def extract_patches_from_dir(directory, patchsize=230,
                             step=115,
                             ):
    """
    Extract patches from an entire directory of images.
    """
    output = {}
    imageList = []
    for fname in os.listdir(directory):
        if fname[-4:] == '.png':
            outname = fname.split('.')[0]
            imageList.append(outname.split('_label')[0])
    imageList = set(imageList)
    for i in imageList:
        imageSet=[]
        imageName = os.path.join(directory,i+'.png')
        image = io.imread(imageName,as_grey=True)
        imgW,imgH = image.shape
        patchsize = int(imgW/10)
        step = int(patchsize/2)      
        labelName = os.path.join(directory,i+'_label.png')
        label = io.imread(labelName,as_grey=True)
        imagePatch= util.view_as_windows(image,window_shape=(patchsize,patchsize),step=step)
        labelPatch= util.view_as_windows(label,window_shape=(patchsize,patchsize),step=step)
        imageSet.append(imagePatch)
        imageSet.append(labelPatch)
        output[i]=np.array(imageSet)
    return output


def save_patches(output):
    for k,v in output.items():
        image = v[0]
        label = v[1]
        weightArray = image.shape[0]
        heightArray = image.shape[1]
        for i in range(heightArray):
            for j in range(weightArray):
                if np.sum(label[j][i]>=1) > 3645:
                    labelName = '_NG'
                    imageName = k + '_%s'%j+'_%s'%i+labelName+'.jpg'
                    imageSavePath = os.path.join('patch_set5','NG',imageName)
                    io.imsave(imageSavePath,image[j][i])
                
                else:
                    if(abs(np.random.normal(0,1))<0.3):
                        labelName = '_OK'
                        imageName = k + '_%s'%j+'_%s'%i+labelName+'.jpg'
                        imageSavePath = os.path.join('patch_set5','OK',imageName)
                        io.imsave(imageSavePath,image[j][i])

def save_patches1(output):
    for k,v in output.items():
        image = v[0]
        label = v[1]
        weightArray = image.shape[0]
        heightArray = image.shape[1]
        for i in range(heightArray):
            for j in range(weightArray):
                if np.sum(label[j][i]) > 3645:
                    labelName = '_label'
                    imageName = k + '_%s'%j+'_%s'%i+'_NG'+'.png'
                    labelName = k + '_%s'%j+'_%s'%i+'_NG'+labelName+'.png'
                    imageSavePath = os.path.join('patchesTest','NG',imageName)
                    labelSavePath = os.path.join('patchesTest','NG',labelName)
                    io.imsave(imageSavePath,image[j][i])
                    io.imsave(labelSavePath,label[j][i])
#                elif abs(np.random.normal(0,1))<0.5 :
#                    labelName = '_label'
#                    imageName = k + '_%s'%j+'_%s'%i+'_OK'+'.png'
#                    labelName = k + '_%s'%j+'_%s'%i+'_OK'+labelName+'.png'
#                    imageSavePath = os.path.join('patchesTest','OK',imageName)
#                    labelSavePath = os.path.join('patchesTest','OK',labelName)
#                    io.imsave(imageSavePath,image[j][i])
#                    io.imsave(labelSavePath,label[j][i])

patches = extract_patches_from_dir('Image6')
save_patches(patches)

#for i in range(10):
#    print(abs(np.random.normal(0,1))<1)