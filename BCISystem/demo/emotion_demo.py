import numpy as np

from data_utils import get_database
from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet


def test():
    data, label = get_database("../database", 5, 3)
    num_classes = 2
    estimator = EEGNet(32, 5000, num_classes)
    estimator.initialize()
    estimator.load_params(f_params="checkpoints/2257375589664/params.pt",
                          f_optimizer="checkpoints/2257375589664/optimizer.pt",
                          f_history="checkpoints/2257375589664/history.json")
    result = estimator.predict(data)
    acc = (np.sum(result == label) / len(label))*100
    print("主观意愿识别准确率为:{:.2f}%".format(acc))


test()
