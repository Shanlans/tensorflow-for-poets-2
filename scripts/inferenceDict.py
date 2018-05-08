# -*- coding: utf-8 -*-

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def get_files(file_dir):
    image_list=[]       
    for file in os.listdir(file_dir):
            image_list.append(os.path.join(file_dir,file))
    return image_list

        
def read_tensor_from_image_file(image_list, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  batch_size = len(image_list)
  image_Name = [i.split('\\')[1].split('.')[0] for i in image_list]
  image_list = tf.cast(image_list,tf.string)  
  input_queue = tf.train.slice_input_producer([image_list],shuffle=False)
  file_reader = tf.read_file(input_queue[0], input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
#  resized = tf.squeeze(resized)
#  normalized = tf.image.per_image_standardization(resized)
#  mul_image = tf.expand_dims(mul_image,0)
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
#  imageSqueeze = tf.squeeze(normalized,axis=0)
  
  image_batch =  tf.train.batch([normalized],
                                batch_size= batch_size,
                                num_threads= 1, 
                                capacity = 2000)
  
#  sess = tf.Session()
#  result = sess.run(image_batch)

  return image_batch,image_Name

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--dict", help="inference a dict of image")
  parser.add_argument("--patch_size",type=int,help="patch size")
  parser.add_argument("--image_size",type=int,help="image size before patching")
  parser.add_argument("--stride",type=int,help="patch stride")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.dict:
    dict_name = args.dict
  if args.patch_size:
    patch_size = args.patch_size
  if args.image_size:
    image_size = args.image_size
  if args.stride:
    stride = args.stride
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  keepprob = "import/input_1/Keep_prob"
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  keep_prob = graph.get_operation_by_name(keepprob)
  
  probArraySize = int((image_size-patch_size)/stride+1)
  probArray = np.zeros((probArraySize,probArraySize),dtype=float)
  imageArray = np.zeros((image_size,image_size),dtype=float)
  onesImage = np.zeros((image_size,image_size),dtype=int)
  onesPatch = np.ones((patch_size,patch_size),dtype=int)
  for i in range(probArraySize):
      for j in range(probArraySize):
          widthStart = i*stride
          widthEnd = widthStart+patch_size
          heightStart = j*stride
          heightEnd = heightStart+patch_size
          
          onesImage[widthStart:widthEnd,heightStart:heightEnd]+= onesPatch   
          
      
  image_batch_op,image_name = read_tensor_from_image_file(get_files(dict_name),
                                                          input_height=input_height,
                                                          input_width=input_width,
                                                          input_mean=input_mean,
                                                          input_std=input_std) 
  
  totalTime = 0
  imageTxtFileName = image_name[0].split('_')[1]
  try:
      sess1 = tf.Session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess1, coord=coord)
      image_batch = sess1.run(image_batch_op)
          
      with tf.Session(graph=graph) as sess:
          for i in range(probArraySize):
              for j in range(probArraySize):
                  num = i*probArraySize+j
                  imageName=image_name[num]
                  probListX=int(imageName.split('_')[2])
                  probListY=int(imageName.split('_')[3])
                  imageIinput = image_batch[num].reshape(1,input_height,input_width,3)
                  start=time.time()
                  results = sess.run(output_operation.outputs[0],
                                     {input_operation.outputs[0]: imageIinput,keep_prob.outputs[0]:1.0})
                  end=time.time()
                  
                  if i==0 and j==0:
                      pass
                  else:
                      totalTime+=(end-start)
                  results = np.squeeze(results)
                  probArray[probListX][probListY]=results[0]
          for i in range(probArraySize):
              for j in range(probArraySize):
                  widthStart = i*stride
                  widthEnd = widthStart+patch_size
                  heightStart = j*stride
                  heightEnd = heightStart+patch_size          
                  imageArray[widthStart:widthEnd,heightStart:heightEnd]+= probArray[i][j] 
          print('\nEvaluation time (1-image): {:.5f}s\n'.format(totalTime/(probArraySize*probArraySize-1)))
          heatMap = np.divide(imageArray,onesImage)
          np.savetxt(os.path.join('tf_files',imageTxtFileName),heatMap)
          sns.set()
          ax = sns.heatmap(heatMap, vmin=0, vmax=1)
          plt.show()
  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      coord.request_stop()      
  coord.join(threads)
  sess1.close()
      
##        for file in os.listdir(dict_name):
##            file_name = os.path.join(dict_name,file)
##            probListX=int(file.split('_')[2])
##            probListY=int(file.split('_')[3])
#            
#      
#         
#            start = time.time()
#            results = sess.run(output_operation.outputs[0],
#                               {input_operation.outputs[0]: t})
#            end=time.time()
#            results = np.squeeze(results)
#            probArray[probListX][probListY]=results[0]
#    
#            top_k = results.argsort()[-5:][::-1]
#            labels = load_labels(label_file)
#
##            print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
##            for i in top_k:
##                print(labels[i], results[i])
#  for i in range(probArraySize):
#        for j in range(probArraySize):
#            widthStart = i*stride
#            widthEnd = widthStart+patch_size
#            heightStart = j*stride
#            heightEnd = heightStart+patch_size          
#            imageArray[widthStart:widthEnd,heightStart:heightEnd]+= probArray[i][j] 
#    
#  heatMap = np.divide(imageArray,onesImage)
#  import seaborn as sns
#  import matplotlib.pyplot as plt
#  sns.set()
#  ax = sns.heatmap(heatMap, vmin=0, vmax=1)
#  plt.show()
            