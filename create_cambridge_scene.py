import os
import json

# Train and Test file paths for a particular Scene

train_gt_files = 'ShopFacade/dataset_train.txt' #Specify path to files for training
test_gt_files = 'ShopFacade/dataset_test.txt' #Specify path to files for testing

f = open(train_gt_files,'r')
train_gt_files = f.readlines()

f = open(test_gt_files,'r')
test_gt_files = f.readlines()

train_gt_files = train_gt_files[3:]
test_gt_files = test_gt_files[3:]

dataset = [[],[],[],[]]

for i in range(len(train_gt_files)):
	temp = train_gt_files[i].split(' ')
	temp = [j.strip() for j in temp]
	dataset[0].append(temp[0])
	temp2 = temp[4:]
	temp2 = [float(j) for j in temp2]
	temp = temp[1:3]
	temp = [float(j) for j in temp]
	dataset[1].append(temp)
	dataset[2].append(temp2)
	dataset[3].append(temp)

# Dump dataset for training (specify path)

with open('ShopFacade/dataset_train_mod.txt','w') as f:
	json.dump(dataset,f)


dataset = [[],[],[],[]]

for i in range(len(test_gt_files)):
	temp = test_gt_files[i].split(' ')
	temp = [j.strip() for j in temp]
	dataset[0].append(temp[0])
	temp2 = temp[4:]
	temp2 = [float(j) for j in temp2]
	temp = temp[1:3]
	temp = [float(j) for j in temp]
	dataset[1].append(temp)
	dataset[2].append(temp2)
	dataset[3].append(temp)

# Dump dataset for testing (specify path)

with open('ShopFacade/dataset_test_mod.txt','w') as f:
	json.dump(dataset,f)
