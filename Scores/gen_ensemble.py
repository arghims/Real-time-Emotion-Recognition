import random
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



model_dict = [
    #('resnet18', 'resnet18_rot30_2019Nov05_17.44'),
    #('resnet50_pretrained_vgg', 'resnet50_pretrained_vgg_rot30_2019Nov13_08.20'),
    #('resnet101', 'resnet101_rot30_2019Nov14_18.12'),
    #('cbam_resnet50', 'cbam_resnet50_rot30_2019Nov15_12.40'),
    #('efficientnet_b2b', 'efficientnet_b2b_rot30_2019Nov15_20.02'),
    #('resmasking_dropout1', 'resmasking_dropout1_rot30_2019Nov17_14.33'),
    #('resmasking', 'resmasking_rot30_2019Nov14_04.38'),
    ('resnet18', 'resnet18__n_2020Jun27_09.51'),
#     ('resnet50_pretrained_vgg', 'resnet50_pretrained_vgg_rot30_2019Nov13_08.20'),
    ('resnet101', 'resnet101__n_2020Jun27_09.16'),
     ('cbam_resnet50', 'cbam_resnet50__n_2020Jun27_09.40'),
     ('efficientnet_b2b', 'efficientnet_b2b__n_2020Jun27_10.00'),
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


# model_dict_proba_list = list(map(list, product([0, 1], repeat=len(model_dict))))

model_dict_proba_list = [[1, 1, 1, 1, 1, 1, 1]]


def main():
    test_results_list = []
    for model_name, checkpoint_path in model_dict:
        test_results = np.load('./saved/results/{}.npy'.format(checkpoint_path), allow_pickle=True)
        test_results_list.append(test_results)
    test_results_list = np.array(test_results_list)

    # load test targets
    test_targets = np.load('./saved/targets/jaffe_target.npy', allow_pickle=True)

    model_dict_proba = [1,1,1,1]

    tmp_test_result_list = []
    for idx in range(len(model_dict_proba)):
        tmp_test_result_list.append(model_dict_proba[idx] * test_results_list[idx])
    tmp_test_result_list = np.array(tmp_test_result_list)
    tmp_test_result_list = np.sum(tmp_test_result_list, axis=0)
    tmp_test_result_list = np.argmax(tmp_test_result_list, axis=1)

    correct = np.sum(np.equal(tmp_test_result_list, test_targets))

    acc = (correct / 70) * 100
    print(acc)


if __name__ == "__main__":
    main()
