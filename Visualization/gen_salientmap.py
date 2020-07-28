import matplotlib.pyplot as plt
import os
import random
import json
import imgaug
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

import models
from utils.generals import make_batch
from torchvision.transforms import transforms

EMOTION_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(model_name,idx):	
	model = getattr(models, model_name)
	model = model(in_channels=3, num_classes=7)
	state = torch.load('/home/aditya/Downloads/checkpoints2/cbam_resnet50__n_2020Jun24_13.32')
	model.load_state_dict(state['net'])
	model.cuda()
	model.eval()
	
	test_set = pd.read_csv('saved/data/jaffe/test.csv')
	path_list = test_set['filepath'].to_list()
	emotions = test_set['emotions'].to_list()
	transform = transforms.Compose([
		    transforms.ToPILImage(),
		    transforms.ToTensor(),
		])
	image = cv2.imread('saved/data/jaffe/'+path_list[idx])
	face_cascade = cv2.CascadeClassifier('saved/xml/haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
		r = max(w, h) / 2
		centerx = x + w / 2
		centery = y + h / 2
		nx = int(centerx - r)
		ny = int(centery - r)
		nr = int(r * 2)
		image = image[ny:ny+nr, nx:nx+nr]
	pix_val = cv2.resize(image,(224,224))[:,:,0]
	#covering parts of image
	size = [range(0,56),range(56,112),range(112,168),range(168,224)]
	l = len(size)
	flag =  1
	ip = np.zeros((1,224,224,1))
	for i in range(l):
		for j in range(l):
			test = np.copy(pix_val)
			for r in size[i]:
				for c in size[j]:
					test[r][c] = 0
			test = test.reshape(1,224,224,1)
			test = test.reshape(224,224)
			test = np.dstack([test]*3)
			test_orig = test
			with torch.no_grad():
				test = transform(test)
				test = make_batch(test)
				test = test.cuda(non_blocking=True)
				outputs = model(test).cpu()
				outputs = F.softmax(outputs, 1)
			plt.imshow(test_orig, interpolation='nearest')
			plt.show()
			print('Predicted class = ',EMOTION_DICT[torch.argmax(outputs).item()],' with probability = ',outputs[0][torch.argmax(outputs).item()].item())
			if flag == 1:
				ip = np.copy(test_orig)
				flag = 0
			else:
				ip = np.concatenate([ip, test_orig])
	torch.cuda.empty_cache()
      
if __name__ == '__main__':
	main('cbam_resnet50',9)
