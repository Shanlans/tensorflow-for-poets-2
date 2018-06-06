# -*- coding: utf-8 -*-
import os
import shutil

def copyfiles(path):
    for roots,folder,files in os.walk(path):
        for i in files:
            if 'NG' in i:
                srcpath = os.path.join(roots,i)
                dstpath = os.path.join('D:\\pythonworkspace\\tensorflow-for-poets-2\\tf_files\\patch_set1\\NG',i)                
            elif 'OK' in i:
                srcpath = os.path.join(roots,i)
                dstpath = os.path.join('D:\\pythonworkspace\\tensorflow-for-poets-2\\tf_files\\patch_set1\\OK',i)
            shutil.copy(srcpath,dstpath)

copyfiles('patch_set3')