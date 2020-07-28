import models
import numpy as np
import torch.multiprocessing as mp
import torch
import imgaug
import os
import json
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs['cwd'] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    from trainers.tta_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs['distributed'] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    return models.__dict__[configs['arch']]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.jaffedataset import jaffe

    # todo: add transform
    train_set = jaffe('train', configs)
    val_set = jaffe('val', configs)
    test_set = jaffe('test', configs, tta=True, tta_size=10)
    return train_set, val_set, test_set


if __name__ == "__main__":
    main('./configs/jaffe_config.json')
