# -*- coding: utf-8 -*-

from PIL import Image
import os

path = 'patches/NG'

for f in os.listdir(path):
    fpath = os.path.join(path,f) 
    im = Image.open(fpath)
    rgb_im = im.convert('RGB')
    fname = f.split('.')[0] + '.jpg'
    rgb_im.save(os.path.join('patches',fname))