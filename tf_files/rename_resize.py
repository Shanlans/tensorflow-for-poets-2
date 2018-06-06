# -*- coding: utf-8 -*-
import os
import cv2

path ='Image5/Mask'

count = 10000
#for i in os.listdir(path):
#    fileName = i
#    name = i.split('_')
#    os.rename(os.path.join(path,fileName),os.path.join(path,name[0]+'_'+name[2]))

#for i in os.listdir(path):
#    fileName = i
#    name = i.split('_')
#    os.rename(os.path.join(path,fileName),os.path.join(path,name[0]+'_'+str(count)+'_label.png'))
#    count+=1


for i in os.listdir(path):
    fileName = i
    print(fileName)
    name = i.split('_')
    os.rename(os.path.join(path,fileName),os.path.join(path,'ImageA16S16A_'+str(count)+'_label.png'))
    count+=1

#for i in os.listdir(path):
#    fileName = i
#    newName = fileName.split('_')[1] + '.png'
#    image = cv2.imread(os.path.join(path,i), 0)
#    result = cv2.resize(image, (2700,2700), interpolation=cv2.INTER_LINEAR)
#    cv2.imwrite(os.path.join(path,i), result)


