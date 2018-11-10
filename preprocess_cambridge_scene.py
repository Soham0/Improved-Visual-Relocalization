import json
import os
import numpy as np 

train_gt_files = 'ShopFacade/' #Path to ground truth for particular scene
test_gt_files = 'ShopFacade/' #Path to ground truth for a particular scene

f = open(train_gt_files + 'dataset_train_mod.txt','r')
train = json.load(f)

f = open(test_gt_files + 'dataset_test_mod.txt','r')
test = json.load(f)

train[0] = [train_gt_files + train[0][i] for i in range(len(train[0]))]
test[0] = [test_gt_files + test[0][i] for i in range(len(test[0]))]

ids = range(0,len(train[0]),10)

#Precompute anchors

anchors = [[],[]]

for i in ids:
	anchors[0].append(train[0][i])
	anchors[1].append(train[1][i])

# Save anchors

with open(train_gt_files + 'anchors.txt','w') as f:
	json.dump(anchors, f)

# Create and dump train, test data in the format for model input

for i in range(len(train[0])):
	train[1][i] = (np.array(train[1][i]) - np.array(anchors[1]))
	train[1][i] = list(train[1][i])
	train[1][i] = [list(j) for j in train[1][i]]
	train[1][i] = sum(train[1][i],[])

with open(train_gt_files + 'traindata.txt','w') as f:
	json.dump(train, f)

# train[1] = sum(train[1],[])

for i in range(len(test[0])):
	test[1][i] = (np.array(test[1][i]) - np.array(anchors[1]))
	test[1][i] = list(test[1][i])
	test[1][i] = [list(j) for j in test[1][i]]
	test[1][i] = sum(test[1][i],[])

with open(test_gt_files + 'testdata.txt','w') as f:
	json.dump(test, f)

# test[1] = sum(test[1],[])
