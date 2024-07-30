import sys
import time

import threading

import copy
import random

import numpy as np
import torch
import yaml
from metabci.brainflow.communicate import Communication
from metabci.brainda.algorithms.identity_recognition.infer import Inference
from metabci.brainflow.amplifiers import VirtualAmplifier


class Calculate(threading.Thread):
    def __init__(self):
        super().__init__()
        config_path = "identity_recognition/config/config_mat.yaml"
        cfg_file = open(config_path, 'r')
        cfg = yaml.safe_load(cfg_file)
        self.inference = Inference(cfg)
        self.amplifier = VirtualAmplifier(5, 32, 1000)

    def run(self):
        while (1):
            if self.amplifier.get_data_len() >= 5:
                data = self.amplifier.get_samples()
                data = data.transpose(1, 0, 2)
                data = data.reshape(32, 5000)
                print("开始计算")
                result = self.inference.infer(data, True)
                print("计算结束")
                self.amplifier.back_result(result)
            time.sleep(2)
