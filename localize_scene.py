import os
import numpy as np
from skimage import io, transform
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import random
import json
import math
import cPickle
import matplotlib.pyplot as plt
import torchvision.models as models

#------------------------------------------------------------------------------------------------------------
#Override Required Functions

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, dof, orig = sample['image'], sample['coordinates'], sample['dof'], sample['origxy']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')

        return {'image': img, 'coordinates': label, 'dof' : dof, 'origxy': orig}

class Resize(object):
	''' Resize the image'''
	def __init__(self, output_size):
	    assert isinstance(output_size, (int, tuple))
	    if isinstance(output_size, int):
	        self.output_size = (output_size, output_size)
	    else:
	        assert len(output_size) == 2
	        self.output_size = output_size

	def __call__(self, sample):
	    image, label, dof, orig = sample['image'], sample['coordinates'], sample['dof'], sample['origxy']
	    h, w = image.shape[:2]
	    new_h, new_w = self.output_size

	    img = transform.resize(image, (new_h, new_w), mode='constant')

	    return {'image': img, 'coordinates': label, 'dof': dof, 'origxy':orig}

    
class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
	    output_size (tuple or int): Desired output size. If int, square crop
	        is made.
	"""

	def __init__(self, output_size):
	    assert isinstance(output_size, (int, tuple))
	    if isinstance(output_size, int):
	        self.output_size = (output_size, output_size)
	    else:
	        assert len(output_size) == 2
	        self.output_size = output_size

	def __call__(self, sample):
	    image, label, dof, orig = sample['image'], sample['coordinates'], sample['dof'], sample['origxy']
	    h, w = image.shape[:2]
	    new_h, new_w = self.output_size
	    
	    top = np.random.randint(0, h - new_h)
	    left = np.random.randint(0, w - new_w)

	    image = image[top: top + new_h,
	                  left: left + new_w]



	    return {'image': image, 'coordinates': label, 'dof': dof, 'origxy':orig}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, dof, orig = sample['image'], sample['coordinates'], sample['dof'], sample['origxy']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image, (2, 0, 1))
        # image = image[np.newaxis,:,:]
        return {'image': torch.from_numpy(image).float(), 'coordinates': torch.from_numpy(np.array([label])).float(), 'dof': torch.from_numpy(np.array([dof])).float(), 'origxy': torch.from_numpy(np.array([orig])).float()}

#-------------------------------------------------------------------------------------------------------------

class Cambridge(torch.utils.data.Dataset):
    
    def __init__(self, gt_file, dataset_root,anchors, shuffle = False, transform = None):
        
        self.dataset_root = dataset_root
        # self.all_file_names = [dataset_root + x for x in os.listdir(dataset_root)]
        
        with open(gt_file) as f:
        	temp_gt = json.load(f)

        # with open(anchors) as g:
        # 	temp_anchors = json.load(g)

        self.all_file_names = temp_gt[0]
        self.labels = temp_gt[1]
        self.dof = temp_gt[2]
        self.orig = temp_gt[3]
        # self.anch = temp_anchors[1]

        # self.labels = [sum(temp_gt[1][i],[]) for i in range(len(temp_gt[1]))]

        self.shuffle = shuffle
        
        if self.shuffle:
            temp = list(zip(self.all_file_names, self.labels, self.dof, self.orig))
            random.shuffle(temp)
            self.all_file_names, self.labels, self.dof, self.orig = zip(*temp)
        
        self.transform = transform
    
    
    def __len__(self):
        return len(self.all_file_names)

        
    def __getitem__(self, idx):
        
        # img_name = os.path.join(self.dataset_root, self.all_file_names[idx])
        image = io.imread(self.all_file_names[idx])
        sample = {'image': image, 'coordinates': self.labels[idx], 'dof' : self.dof[idx], 'origxy': self.orig[idx]}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

batch_size = 16

# composed = transforms.Compose([Rescale(256), RandomCrop(224),ToTensor()])
composed = transforms.Compose([Resize(224),ToTensor()])

train_dataset = Cambridge('ShopFacade/traindata.txt', 'ShopFacade/', 'ShopFacade/anchors.txt', shuffle = True, transform=composed)
test_dataset = Cambridge('ShopFacade/testdata.txt', 'ShopFacade/', 'ShopFacade/anchors.txt', shuffle = True, transform=composed)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

#printing the no. of training samples
print len(train_dataset)
print len(test_dataset)

# for i in range(len(train_dataset)):
#     sample = train_dataset[i]

#     print(i, sample['image'].shape)

#	----------------------------------------MODEL DEFINITION------------------------------------

model = models.densenet161(pretrained=True)
# model = models.inception_v3(pretrained=True)

GAP = nn.AvgPool2d(7)

dropout = nn.Dropout(p=0.6)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feat_extractor = model.features
        # self.feat_extractor = model.Conv2d_1a_3x3
        self.classifier = nn.Linear(2208, 33)
        self.regressor = nn.Linear(2208, 66)
        self.dof_regressor = nn.Linear(2208,4)
        
    def forward(self, x):
        out = self.feat_extractor(x)
        out = GAP(out)
        out = out.view(out.size(0), -1)
        classify = self.classifier(F.relu(out))
        regress = self.regressor(F.relu(out))
        dof_regress = self.dof_regressor(F.relu(out))
        return classify,regress,dof_regress


net = Net()

################### TRAINING ####################

net = net.cuda()
criterion = nn.CrossEntropyLoss()
klcriterion = nn.KLDivLoss()
mse = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0003)


numEpochs = 300


softmax = nn.Softmax()

def custom_loss(classify, regress, labels, class_labels, dof, dof_regress):
	dist = regress - labels
	dist = dist**2
	

	loss = 0.0

	for i in range(dist.size()[0]):
		for j in range(0,66,2):
			loss += (dist[i][j] + dist[i][j+1])*classify[i][int(j/2)]

	_,class_labels = torch.min(class_labels,1)
	_,pred_labels = torch.min(classify,1)

	
	
	loss = 2.4*loss + 0*criterion(classify, class_labels) + 0.5*mse(dof_regress, dof)

	return loss

scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

with open('ShopFacade/anchors.txt', 'r') as f:
	anchors = json.load(f)

for epoch in range(numEpochs):  # loop over the dataset multiple times

	total_loss = 0
	train_accuracy = []
	train_distance = []
	median_train_distance = []
	train_dof_accuracy = []

	test_accuracy = []
	test_distance = []
	median_test_distance = []
	test_dof_accuracy = []

	temp_dist_test = []
	temp_dist_gt = []

	net.train()

	for i, data in enumerate(train_dataloader):
	    
		# get the inputs
		images, labels, dof, origxy = data['image'], data['coordinates'], data['dof'], data['origxy']
		labels = labels.view(-1,66)
		dof = dof.view(-1,4)
		temp_class_labels = labels**2
		class_labels = torch.FloatTensor(temp_class_labels.size()[0],33)
		for j in range(class_labels.size()[0]):
			for k in range(class_labels.size()[1]):
				class_labels[j][k] = temp_class_labels[j][2*k] + temp_class_labels[j][2*k + 1]
		
		images = images.cuda()
		labels = labels.cuda()
		dof = dof.cuda()
		class_labels = class_labels.cuda()

		temp_labels = labels
		temp_dof = dof
		images = Variable(images)
		labels = Variable(labels)
		class_labels = Variable(class_labels)
		dof = Variable(dof)

		optimizer.zero_grad()
		classify,regress,dof_regress = net(images)
		classify = softmax(classify)
		loss = custom_loss(classify, regress, labels,class_labels, dof,dof_regress)
		loss.backward()
		optimizer.step()

		total_loss += loss.data[0]
		_, predicted = torch.max(classify.data, 1)

		correct = 0
		correct_dof = 0
		diff = torch.abs(regress - labels)

		temp_dist = 0

		for j in range(predicted.size()[0]):
			if ((diff[j][2*predicted[j]].data.cpu().numpy()**2 + diff[j][2*predicted[j]+1].data.cpu().numpy()**2)**0.5 < 2):
				correct+=1
			temp_dist += (diff[j][2*predicted[j]].data.cpu().numpy()**2 + diff[j][2*predicted[j]+1].data.cpu().numpy()**2)**0.5
			median_train_distance.append((diff[j][2*predicted[j]].data.cpu().numpy()**2 + diff[j][2*predicted[j]+1].data.cpu().numpy()**2)**0.5)

		dof_distances = torch.abs(dof_regress-dof)
		predicted_dof,_ = torch.max(dof_distances.data,1)

		for j in range(predicted_dof.size()[0]):
			if (predicted_dof[j] < 0.3):
				correct_dof +=1

		train_dof_accuracy.append(float(correct_dof)/predicted_dof.size()[0])
		train_accuracy.append(float(correct)/predicted.size()[0])
		train_distance.append(float(temp_dist)/predicted.size()[0])

	median_train_distance = np.array(median_train_distance)

	print('Epoch: %d, Training Loss: %.4f, Training Acc: %.4f , Train Mean Dist: %.4f , Train DOF Acc: %.4f, Train Median Dist: %.4f' %(epoch+1, total_loss, (sum(train_accuracy)/float(len(train_accuracy))), (sum(train_distance)/float(len(train_distance))),(sum(train_dof_accuracy)/float(len(train_dof_accuracy))), np.median(median_train_distance)))

	if((epoch+1)% 5 == 0):
		net.eval()

		for k, data in enumerate(test_dataloader):

			images, labels, dof, origxy = data['image'], data['coordinates'], data['dof'], data['origxy']
			labels = labels.view(-1,66)
			dof = dof.view(-1,4)

			images = images.cuda()
			labels = labels.cuda()
			dof = dof.cuda()

			images = Variable(images)
			labels = Variable(labels)
			dof = Variable(dof)

			classify,regress,dof_regress = net(images)
			classify = softmax(classify)

			_, predicted = torch.max(classify.data, 1)

			correct = 0
			correct_dof = 0
			diff = torch.abs(regress - labels)


			for j in range(predicted.size()[0]):
				if ((diff[j][2*predicted[j]].data.cpu().numpy()**2 + diff[j][2*predicted[j]+1].data.cpu().numpy()**2)**0.5 < 2):
					correct+=1
				temp_dist_test.append([diff[j][2*predicted[j]].data.cpu().numpy() + anchors[1][predicted[j]][0], diff[j][2*predicted[j]+1].data.cpu().numpy() + anchors[1][predicted[j]][1]])
				temp_dist_gt.append([origxy[j].numpy()])
				median_test_distance.append((diff[j][2*predicted[j]].data.cpu().numpy()**2 + diff[j][2*predicted[j]+1].data.cpu().numpy()**2)**0.5)

			dof_distances = torch.abs(dof_regress-dof)
			predicted_dof,_ = torch.max(dof_distances.data,1)

			for j in range(predicted_dof.size()[0]):
				if (predicted_dof[j] < 0.3):
					correct_dof +=1

			test_dof_accuracy.append(float(correct_dof)/predicted_dof.size()[0])
			test_accuracy.append(float(correct)/predicted.size()[0])
			test_distance.append(float(temp_dist)/predicted.size()[0])

		test_errors = median_test_distance
		median_test_distance = np.array(median_test_distance)

		test_errors = [list(k) for k in test_errors]
		test_errors = list(test_errors)
		
		'''
		Saving test distances and ground truths for x,y for calculating test accuracy

		'''

		with open('ShopFacade/test_errors.txt','w') as f:
			cPickle.dump(test_errors,f)

		with open('ShopFacade/pred_dist.txt','w') as f:
			cPickle.dump(temp_dist_test,f)

		with open('ShopFacade/gt_dist.txt','w') as f:
			cPickle.dump(temp_dist_gt,f)

		print('Epoch: %d, Acc: %.4f , Mean Dist: %.4f, DOF Acc: %.4f, Median Dist: %.4f' %(epoch+1, (sum(test_accuracy)/float(len(test_accuracy))), (sum(test_distance)/float(len(test_distance))), (sum(test_dof_accuracy)/float(len(test_dof_accuracy))), np.median(median_test_distance)))
		
