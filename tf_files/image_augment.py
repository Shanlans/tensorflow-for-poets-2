# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image,ImageEnhance


def image_augment(filename,angle=None,luminance=None,Contrast=None):
    #img = Image.open(filename)
    img = cv2.imread(filename)
    name = filename.split('_')
    path = os.getcwd()
    currentdir = path.split('\\')
    paraent_path = os.path.dirname(path)  
        
    if angle is not None:
        ropath = os.path.join(paraent_path,currentdir[-1]+'_rotate')
        rodirsuccess = mkdir(ropath)
        for a in angle:
#            img = img.rotate(a)
            rows,cols,channels = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),a,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            rotation_name = name[0]+'_ro'+str(a)
            for n in name[1:]:
                rotation_name+=('_'+n)         
                save_path = os.path.join(ropath,rotation_name)
            try:
#                img.save(save_path)
                cv2.imwrite(save_path,dst)
            except IOError:
                print("cannot rotation")
    else:
        #print("Don't do any rotation")
        pass
    
    if luminance is not None:
        lupath = os.path.join(paraent_path,currentdir[-1]+'_luminance')
        ludirsuccess = mkdir(lupath)
        for l in luminance:
            img = ImageEnhance.Brightness(img).enhance(l)
            luminance_name = name[0]+'_lu'+str(l)
            for n in name[1:]:
                luminance_name+=('_'+n)         
                save_path = os.path.join(lupath,luminance_name)
            try:
                img.save(save_path)
            except IOError:
                print("cannot enhance luminance")
    else:
        #print("Don't do any lumiance enhancement")
        pass
        
    if Contrast is not None:
        copath = os.path.join(paraent_path,currentdir[-1]+'_contrast')
        codirsuccess = mkdir(copath)
        for C in Contrast:
            img = ImageEnhance.Contrast(img).enhance(C)
            contrast_name = name[0]+'_co'+str(C)
            for n in name[1:]:
                contrast_name+=('_'+n)         
                save_path = os.path.join(copath,contrast_name)
            try:
                img.save(save_path)
            except IOError:
                print("cannot enhance contrast")
    else:
        #print("Don't do any contrast enhancement")
        pass
        
def flip_image(dirpath):
    for filename in os.listdir(dirpath):
        imageName = os.path.join(dirpath,filename)
        name = filename.split('_')
        img = cv2.imread(imageName)
        print(imageName)
        horizontal_img = img.copy()
        vertical_img = img.copy()
        both_img = img.copy()
        horizontal_img = cv2.flip( img, 0 )
        horizontal_img = cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2GRAY)
        vertical_img = cv2.flip( img, 1 )
        vertical_img = cv2.cvtColor(vertical_img, cv2.COLOR_BGR2GRAY)
        both_img = cv2.flip( img, -1 )
        both_img = cv2.cvtColor(both_img, cv2.COLOR_BGR2GRAY)
        if len(name) == 5:
            h_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'h'+'_'+name[4]
            v_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'v'+'_'+name[4]
            b_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'b'+'_'+name[4]
        elif len(name) == 6:
            h_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'h'+'_'+name[4]+'_'+name[5]
            v_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'v'+'_'+name[4]+'_'+name[5]
            b_image_name= name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+'b'+'_'+name[4]+'_'+name[5]
#        dirpath = 'D:\\pythonworkspace\\tensorflow-for-poets-2\\tf_files\\patch_set1\\NG'
        h_save_name=os.path.join(dirpath,h_image_name)
        v_save_name=os.path.join(dirpath,v_image_name)
        b_save_name=os.path.join(dirpath,b_image_name)
        cv2.imwrite(h_save_name,horizontal_img)
        cv2.imwrite(v_save_name,vertical_img)
        cv2.imwrite(b_save_name,both_img)
        
def rotate_image(dirpath):
    for filename in os.listdir(dirpath):
        imageName = os.path.join(dirpath,filename)
        name = filename.split('_')
        img = cv2.imread(imageName)
        rows,cols,channels = img.shape
        a = [90,180,270]
#        dirpath = 'D:\\pythonworkspace\\tensorflow-for-poets-2\\tf_files\\patch_set1\\NG'
        for i in a:
            M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            if len(name) == 6:
                rotation_name = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+name[4]+'_'+str(i)+'_'+name[5]
            else:
                rotation_name = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+str(i)+'_'+name[4]
            cv2.imwrite(os.path.join(dirpath,rotation_name),dst)
            
def luminance_image(dirpath):
    for filename in os.listdir(dirpath):
        imageName = os.path.join(dirpath,filename)
        name = filename.split('_')
        img = Image.open(imageName)
        luminance = [i/10 for i in range(20)[1:][::5][1:]]
        for i in luminance:
            img = ImageEnhance.Brightness(img).enhance(i)
            lum_name = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]+'_'+name[4]+'_'+name[5]+'_'+str(i)+name[6]
            img.save(lum_name)
        
    
#file_dir = 'C:\\Users\\sunny\\Desktop\\Clss\\Zangwu'
##file_dir1 = 'C:\\Users\\sunny\\Desktop\\Clss\\Tuohen\\Tuohenrotate'
##file_dir2 = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\dataset1\\IR_train7_rotate_luminance'
#os.chdir(file_dir)
#
#angle = [i for i in range(360)[::89]]
##luminance = [i/10 for i in range(20)[1:][::5][1:]]
##print(luminance)
#
#print(angle)
##
#for file in os.listdir(file_dir):
#    print(file)
#    if file.split('.')[1] == 'png':
#        print(file)
#        image_augment(file,angle=angle)

#os.chdir(file_dir1)
#
#for file in os.listdir(file_dir1):
#    image_augment(file,angle=None,luminance=luminance)
#
#os.chdir(file_dir2)
#
#for file in os.listdir(file_dir2):
#    image_augment(file,angle=None,luminance=None,Contrast=luminance)
#flip_image('patch_set5/NG')
#print('finish_flip')
rotate_image('patch_set5/NG')
print('finish_rotate')