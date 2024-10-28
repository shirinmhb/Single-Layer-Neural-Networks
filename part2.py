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
    DATADIR="NNDL-Pr1/Alphabet/"
    data = []
    label = []
    temp = DATADIR + str(1)
    l = np.full((20, 26), -1)
    for j in range(20):
      l[j][0] = 1
    data = self.read_image(temp)
    label = l
    for i in range(1,26):
      temp = DATADIR + str(i+1)
      l = np.full((20, 26), -1)
      for j in range(20):
        l[j][i] = 1
      data = np.concatenate((data, self.read_image(temp)), axis=0)
      label = np.concatenate((label, l), axis=0)

    self.data = data
    self.label = label
    # plt.imshow(data[511].reshape(60,60), cmap="gray")
    # plt.show()

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
    for i in range(len(new_img)):
      for j in range(len(new_img[i])):
        if new_img[i][j] != 255:
          new_img[i][j] = new_img[i][j] + 1
    return new_img
  
  def bipolar_sigmoid2(self, y_in):
    print("y_in", y_in)
    b = ((1 - np.exp(-1*y_in))/(1 + np.exp(-1*y_in)))
    print("b", b)
    return b

  def bipolar_sigmoid(self, y_in):
    # print("y_in", y_in)
    b = 2 / (1 + np.exp(-1 * y_in)) - 1
    # print("b", b)
    return b

  def learn(self, trainData, trainLabel):
    #initial weights
    weights = np.random.uniform(low=0, high=0.000001, size=(3600,26))
    # weights = np.full((3600,26), 0.000001)
    b = np.random.uniform(low=0, high=0.000001, size=(26))
    # b = np.full((26), 0.000001)
    alpha = self.learningRate
    largestChange = 100
    while abs(largestChange) > 0:
      # print(largestChange)
      largestChange = 0
      for p in range(len(trainData)): # for each train sample
        y = np.zeros(26)
        for j in range(0,26): # for each y out
          y_in = b[j] 
          y_in += trainData[p].T @ weights[:,j]
          y[j] = self.bipolar_sigmoid(y_in)
          if y[j] != trainLabel[p][j]:  #update weights and bias
            deltaB = alpha * (trainLabel[p][j] - y[j]) * (1 + y[j]**2) 
            if deltaB==0:
              print(deltaB, (trainLabel[p][j] - y[j]), (1 + y[j]**2) )
            b[j] += deltaB
            if abs(deltaB) > largestChange:
              largestChange = deltaB
            for i in range(len(trainData[p])):
              deltaW = alpha * (trainLabel[p][j] - y[j]) * (1 + y[j]**2) * trainData[p][i]
              if deltaW==0:
                print(deltaW, (trainLabel[p][j] - y[j]), (1 + y[j]**2) , trainData[p][i])
              weights[i][j] += deltaW
              if abs(deltaW) > largestChange:
                largestChange = deltaW
      print("l", largestChange)
          # else:
          #   print("sag", p, y[j], trainLabel[p][j])
      # print("largestChange2", largestChange)

    self.weight = weights
    self.b = b

  def predict(self, features):
    weights = self.weight
    b = self.b
    y = np.zeros(26)
    for i in range(0, 26):
      y_in = b[i]
      y_in += (features.T @ weights[:,i])
      y[i] = self.bipolar_sigmoid(y_in)
    return y

  def LOOCV(self):
    correct = 0
    err = 0
    for i in range(len(self.data)):
      testData = self.data[i]
      trainData = np.concatenate((self.data[:i], self.data[i+1:]), axis=0)
      # print(testData.shape, trainData.shape)
      testLabel = self.label[i]
      trainLabel = np.concatenate((self.label[:i], self.label[i+1:]), axis=0)
      self.learn(trainData, trainLabel)
      predLabel = self.predict(testData)
      # print(predLabel)
      # print(testLabel)
      if (predLabel==testLabel).all():
        correct += 1
        print(i)
      else:
        err += ((np.sum((predLabel - testLabel)**2))/len(predLabel))
        print("err", i, err)
    print("err", err/i)
    print("accuracy", correct/i*100)

  def robustness2(self, percent):
    corruptedData = self.data
    numberCorruption = percent * 0.01 * len(corruptedData)
    for p in corruptedData:
      num = 0
      notCorrupted = list(range(0, len(p)))
      while num < numberCorruption:
        r = random.choice(notCorrupted)
        notCorrupted.remove(r) 
        p[r] == 255 - p[r]
        num += 1
    correct = 0
    self.learn(self.data, self.label)
    for i in range(len(corruptedData)):
      y_pred = self.predict(corruptedData[i])
      if (y_pred==self.label[i]).all():
        correct += 1
    print(correct)

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
        if self.data[p][r] < 127:  #-1 blacke
          corruptedData[p][r] = 255 - self.data[p][r]
          num +=1
        if len(notCorrupted) == 0:
          print(num, p)
          break
    correct = 0
    plt.imshow(corruptedData[0].reshape(60,60), cmap="gray")
    plt.show()
    self.learn(self.data, self.label)
    for i in range(len(corruptedData)):
      y_pred = self.predict(corruptedData[i])
      if (y_pred==self.label[i]).all():
        correct += 1
    print("accuracy",percent,"% noise is", correct/len(corruptedData)*100,"%")

perceptron = Perceptron()
perceptron.LOOCV()
# perceptron.robustness(15)
