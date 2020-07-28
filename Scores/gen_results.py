
import os
import random
import json
import imgaug
import torch
import numpy as np

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.jaffedataset import jaffe
from utils.generals import make_batch

model_dict = [ 
    #('resnet18', 'resnet18_rot30_2019Nov05_17.44'),
    #('resnet50_pretrained_vgg', 'resnet50_pretrained_vgg_rot30_2019Nov13_08.20'),
    #('resnet101', 'resnet101_rot30_2019Nov14_18.12'),
    #('cbam_resnet50', 'cbam_resnet50_rot30_2019Nov15_12.40'),
    #('efficientnet_b2b', 'efficientnet_b2b_rot30_2019Nov15_20.02'),
    #('resmasking_dropout1', 'resmasking_dropout1_rot30_2019Nov17_14.33'),
    #('resmasking', 'resmasking_rot30_2019Nov14_04.38'),
#    ('resnet18', 'resnet18__n_2020Jun27_09.51'),
#     ('resnet50_pretrained_vgg', 'resnet50_pretrained_vgg_rot30_2019Nov13_08.20'),
    ('resnet101', 'resnet101__n_2020Jun27_09.16'),
 #    ('cbam_resnet50', 'cbam_resnet50__n_2020Jun27_09.40'),
  #   ('efficientnet_b2b', 'efficientnet_b2b__n_2020Jun27_10.00'),
#     ('resmasking_dropout1', 'resmasking_dropout1__n_2020Jun24_13.09'),
#     ('resmasking', 'resmasking__n_2020Jun24_15.07'),
#     ('alexnet','alexnet__n_2020Jun24_13.57'),
#     ('densenet121','densenet121__n_2020Jun24_13.58'),
#     ('googlenet','googlenet__n_2020Jun24_14.26'),
#     ('inception_resnet_v1','inception_resnet_v1__n_2020Jun24_14.08'),
#     ('resnet152','resnet152__n_2020Jun24_13.26'),
#     ('resnet34','resnet34__n_2020Jun24_13.19'),
#     ('vgg16','vgg16__n_2020Jun24_14.44'),
#     ('vgg19','vgg19__n_2020Jun24_14.15'),
#     ('wide_resnet101_2','wide_resnet101_2__n_2020Jun24_14.23'),
#     ('wide_resnet50_2','wide_resnet50_2__n_2020Jun24_13.48'),
#     ('bam_resnet50','bam_resnet50__n_2020Jun24_16.00')
]


def main():
    with open('./configs/jaffe_config.json') as f:
        configs = json.load(f)
    
    test_set = jaffe('test', configs, tta=True, tta_size=8)

    for model_name, checkpoint_path in model_dict:
        prediction_list = []  # each item is 7-ele array

        print("Processing", checkpoint_path)
        if os.path.exists('./saved/results/{}.npy'.format(checkpoint_path)):
            continue


        model = getattr(models, model_name)
        model = model(in_channels=3, num_classes=7)
         
        state = torch.load(os.path.join('saved/checkpoints', checkpoint_path))
        model.load_state_dict(state['net'])
        
        model.cuda()
        model.eval()

 
        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
                images, targets = test_set[idx]
                images = make_batch(images) 
                images = images.cuda(non_blocking=True)

                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)

        np.save('./saved/results/{}.npy'.format(checkpoint_path), prediction_list) 

        


if __name__ == "__main__":
    main()
