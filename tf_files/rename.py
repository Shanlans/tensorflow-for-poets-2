# -*- coding: utf-8 -*-
import os
import shutil

path ='Image3\\Image'

#for i in os.listdir(path):
#    fileName = i
#    name = i.split('0_mask.png')
#    os.rename(os.path.join(path,fileName),os.path.join(path,name[0]+'label.png'))

dstpath = 'D:\\pythonworkspace\\tensorflow-for-poets-2\\tf_files\\patch_set1\\OK'
for i in os.listdir(path):
    fileName = i
    shutil.copy(os.path.join(path,i),os.path.join(dstpath,i))



