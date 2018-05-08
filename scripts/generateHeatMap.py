# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt


heatMap=np.loadtxt(os.path.join('tf_files','00053'))
#heatMap1=np.loadtxt(os.path.join('tf_files','00000_1'))
#print(np.sum((heatMap==heatMap1)))
heatMap= heatMap*(heatMap>0.8  )
sns.set()
ax = sns.heatmap(heatMap, vmin=0, vmax=1)
plt.show()