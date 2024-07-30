import os
import sys

import numpy as np
import yaml

sys.path.append("../../metabci/brainda/algorithms/identity_recognition")
from metabci.brainda.algorithms.identity_recognition.infer import Inference

config_path = "../config/config_mat.yaml"
cfg_file = open(config_path, 'r')
cfg = yaml.safe_load(cfg_file)

inference = Inference(cfg)


def test():
    database = "../data2"
    files = os.listdir(database)
    if len(files) == 0:
        pass
    result = []
    for subject_name in files:
        all_count = 0
        correct_account = 0
        subject_path = os.path.join(database, subject_name)
        if os.path.isdir(subject_path):
            subject_datas_name = os.listdir(subject_path)
            for subject_data_name in subject_datas_name:
                subject_data_path = os.path.join(subject_path, subject_data_name) + "/data"
                sample_num = len(os.listdir(subject_data_path))
                for i in range(sample_num - 5):
                    data = []
                    with open(os.path.join(subject_path, subject_data_name) + "/label/{}.txt".format(i)) as f:
                        label = int(f.read())
                    for j in range(5):
                        data.append(np.load(subject_data_path + "/{}.npy".format(i + j)))
                    data = np.concatenate(data, axis=-1)
                    identity_map = inference.infer(data)
                    all_count += 1
                    if identity_map["id"] == label:
                        correct_account += 1
            acc = correct_account / (all_count * 1.0) * 100
            result.append((subject_name, acc))
    print(result)


test()
