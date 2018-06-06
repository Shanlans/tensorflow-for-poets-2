# -*- coding: utf-8 -*-
import numpy as np
from skimage import io,util
import os


def extract_patches_from_dir(directory, patchsize=270,
                             step=90,
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
        labelName = os.path.join(directory,i+'_label.png')
        label = io.imread(labelName,as_grey=True)
        list = [[0,2700,0,540],[0,2700,2160,2700],[0,540,540,2160],[2160,2700,540,2160]]
        index = 0
        for j in list:
            image_crop = image[j[0]:j[1],j[2]:j[3]]
            label_crop = label[j[0]:j[1],j[2]:j[3]]
            imagePatch = util.view_as_windows(image_crop,window_shape=(patchsize,patchsize),step=step)
            labelPatch = util.view_as_windows(label_crop,window_shape=(patchsize,patchsize),step=step)
            imagePatch = np.reshape(imagePatch,(-1,patchsize,patchsize))
            labelPatch = np.reshape(labelPatch,(-1,patchsize,patchsize))

            if index == 0:
                imagePatches = imagePatch
                labelPatches = labelPatch
                pass
            else:
                imagePatches = np.append(imagePatches,imagePatch,axis=0)
                labelPatches = np.append(labelPatches,labelPatch,axis=0)
            index+=1

        imageSet.append(imagePatches)
        imageSet.append(labelPatches)
        output[i] = np.array(imageSet)
    return output


def save_patches(output):
    for k,v in output.items():
        image = v[0]
        label = v[1]
        numOfPatches = image.shape[0]
        for i in range(numOfPatches):
            if np.sum(label[i]>=1) > 729:
                labelName = '_NG'
                imageName = k+'_%s'%i+labelName+'.jpg'
                imageSavePath = os.path.join('patch_set5','NG',imageName)
                io.imsave(imageSavePath,image[i])
                
            else:
                if(abs(np.random.normal(0,1))<0.3):
                    labelName = '_OK'
                    imageName = k+'_%s'%i+labelName+'.jpg'
                    imageSavePath = os.path.join('patch_set5','OK',imageName)
                    io.imsave(imageSavePath,image[i])

def save_patches1(output):
    for k,v in output.items():
        image = v[0]
        label = v[1]
        weightArray = image.shape[0]
        heightArray = image.shape[1]
        for i in range(heightArray):
            for j in range(weightArray):
                if np.sum(label[j][i]) > 729:
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

patches = extract_patches_from_dir('shuijing')
save_patches(patches)

#for i in range(10):
#    print(abs(np.random.normal(0,1))<1)