import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from utils.augmenters.augment import seg

EMOTION_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}



class Jaffe(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48):
        self._stage = stage
        self._tta = tta
        self._tta_size = tta_size
        self._configs = configs

        self._image_size = (224, 224)
        self._data = pd.read_csv(
            os.path.join(
                configs['data_path'],
                '{}.csv'.format(stage)
                )
        )

        self._path_list  = self._data['filepath'].tolist()
        self._emotions = pd.get_dummies(self._data['emotions'])
        
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])


    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._path_list)

    def __getitem__(self, idx):
        
        path = self._path_list[idx]
        image = cv2.imread(self._configs['data_path']+'/'+path)
        image = cv2.resize(image, (224,224))

        if self._stage == 'train':
            image = seg(image=image)

        if self._stage == 'test' and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            images = list(map(self._transform, images))
            target = self._emotions.iloc[idx].idxmax()
            return images, target
            
        image = self._transform(image)
        target = self._emotions.iloc[idx].idxmax()
        return image, target


def jaffe(stage, configs=None, tta=False, tta_size=48):
    return Jaffe(stage, configs, tta, tta_size)

