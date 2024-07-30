import numpy as np

from metabci.brainda.algorithms.Blink.Blink import BlinkRecognition
# 输入信号为眼电信号
input = np.random.random(5000)
blink = BlinkRecognition()
blink.start()
for value in input:
    result = blink.recognition(value)
    if result:
        print("用户眨眼")
