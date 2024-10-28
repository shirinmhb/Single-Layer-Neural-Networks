import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import os
import cv2 as cv
import sys
from copy import deepcopy
import random
import collections
np.set_printoptions(threshold=sys.maxsize)

class Perceptron():
  def __init__(self):
    self.learningRate = 1
    self.delta = 0
    DATADIR="NNDL-Pr1/Alphabet/"
    data = []
    label = []
    for i in range(0,26):
      temp = DATADIR + str(i+1)
      l = np.full((20, 26), -1)
      for j in range(20):
        l[j][i] = 1
      data.append(self.read_image(temp))
      label.append(l)

    data = np.array(data)
    label = np.array(label)
    self.data = data.reshape(520, 3600)
    self.label = label.reshape(520, 26)

  def read_image(self, DATADIR):
    path=os.path.join(DATADIR)
    images=[]
    img_size=60
    for img in os.listdir(path):
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            images.append(new_image)
    b=np.array(images)    
    new_img = b.reshape(20,(img_size*img_size))
    bipolar = np.full((20,3600), -1)
    for i in range(len(new_img)):
      for j in range(len(new_img[i])):
        if new_img[i][j] > 127:
          bipolar[i][j] = 1
        else:
          bipolar[i][j] = -1
        # print(bipolar[i][j])
    # plt.imshow(bipolar[0].reshape(60,60), cmap="gray")
    # plt.show()
    # plt.imshow(temp[19].reshape(60,60), cmap="gray")
    # plt.show()
    return bipolar
  
  def learn(self, trainData, trainLabel):
    #initial weights
    weights = np.zeros((3600,26))
    b = np.zeros(26)
    alpha = self.learningRate
    weightChange = True
    epoc = 0
    while weightChange:
      epoc += 1
      # print("epoc", epoc)
      weightChange = False
      for p in range(len(trainData)): # for each train sample
        # s = deepcopy(trainData[p]) 
        # t = deepcopy(trainLabel[p])
        y = np.zeros(26)
        for j in range(0,26): # for each y out
          y_in = b[j] 
          y_in += trainData[p].T @ weights[:,j]
          if y_in > self.delta:
            y[j] = 1
          elif y_in < -1 * self.delta:
            y[j] = -1
          else:
            y[j] = 0

          if y[j] != trainLabel[p][j]: #update weights and bias
            deltaB = alpha * trainLabel[p][j]
            if deltaB != 0:
              weightChange = True
              b[j] += deltaB
            for i in range(len(trainData[p])):
              deltaW = alpha * trainLabel[p][j] * trainData[p][i]
              if deltaW != 0:
                weightChange = True
                weights[i][j] += deltaW
    
    self.weight = weights
    self.b = b

  def predict(self, features):
    weights = self.weight
    b = self.b
    y = np.zeros(26)
    for i in range(26):
      y_in = (features @ weights[:,i].reshape(3600, 1)) + b[i]
      if y_in > self.delta:
        y[i] = 1
      elif y_in < -1 * self.delta:
        y[i] = -1
      else:
        y[i] = 0
    return y

  def LOOCV(self):
    correct = 0
    err = 0
    data = self.data.tolist()
    label = self.label.tolist()
    for i in range(0, len(self.data)):
      testData = data[i]
      trainData = data[:i] + data[i+1:]
      testLabel = label[i]
      trainLabel = label[:i] + label[i+1:]
      self.learn(np.array(trainData), np.array(trainLabel))
      predLabel = self.predict(np.array(testData))
      if (predLabel==testLabel).all():
        correct += 1
        print(i)
      else:
        err += (np.sum((predLabel - testLabel)**2))/len(predLabel)
    print("percent of correct prediction", correct/len(self.data)*100)
    print("error", err/len(self.data))
      
  def robustness(self, percent):
    corruptedData = deepcopy(self.data)
    numOfBlack = 700
    numberCorruption = percent * 0.01 * numOfBlack
    for p in range(len(corruptedData)):
      num = 0
      notCorrupted = list(range(0, len(corruptedData[p])))
      while num < numberCorruption:
        r = random.choice(notCorrupted)
        notCorrupted.remove(r) 
        if self.data[p][r] == -1:  #-1 blacke
          corruptedData[p][r] = 1
          num +=1
        if len(notCorrupted) == 0:
          print(num, p)
          break
    correct = 0
    # plt.imshow(corruptedData[0].reshape(60,60), cmap="gray")
    # plt.show()
    self.learn(self.data, self.label)
    for i in range(len(corruptedData)):
      y_pred = self.predict(corruptedData[i])
      if (y_pred==self.label[i]).all():
        correct += 1
    print("accuracy",percent,"% noise is", correct/len(corruptedData)*100,"%")

perceptron = Perceptron()
# perceptron.LOOCV()
perceptron.robustness(25)
