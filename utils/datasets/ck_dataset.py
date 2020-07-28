import os
import random
import cv2
import numpy as np
from imutils import paths
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from utils.augmenters.augment import seg

random.seed(123)


EMOTION_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'contempt'
}



class CK(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs['image_size'], configs['image_size'])

        self._data = list(paths.list_images(
                configs['data_path']))
        random.shuffle(self._data)

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        data = self._data[idx]
        data = cv2.imread(data)
        image = cv2.resize(data,self._image_size)
        image = image.astype(np.uint8)

        if self._stage == 'train':
            image = seg(image=image)

        if self._stage == 'test' and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            # images = [image for i in range(self._tta_size)]
            images = list(map(self._transform, images))
            target = int(self._data[idx].split(os.path.sep)[-2])
            return images, target

        image = self._transform(image)
        target = int(self._data[idx].split(os.path.sep)[-2])
        return image, target

def ck(stage, configs=None, tta=False, tta_size=48):
	return CK(stage, configs, tta, tta_size)
