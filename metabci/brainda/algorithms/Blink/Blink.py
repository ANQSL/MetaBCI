import enum

import numpy as np


class ThreshlodStatus(enum.Enum):
    InIt = 0
    Normal = 1
    Update = 2


class BlinkRecognition:
    def __init__(self):
        self.count = 0
        self.p = 0
        self.x = np.zeros(1025)

        self.blink_threshold = 0
        self.baseline = 0
        self.eog_threshold = 70

        self.sum = 0
        self.init_count = 200
        self.dynamic_count = 200
        self.discard_count = 200
        self.current_count = 0
        self.running_status = False
        self.threshlod_status = ThreshlodStatus.InIt

    def recognition(self, value):
        if self.running_status:
            self.dynamicThreshlod(value)
            return self.blinkRecognition(value)

        return False

    def start(self):
        self.running_status = True

    def dynamicThreshlod(self, value):
        self.current_count += 1
        if self.threshlod_status == ThreshlodStatus.InIt:
            self.sum += value
            if self.init_count == self.current_count:
                self.baseline = self.sum / self.current_count
                self.blink_threshold = self.baseline + self.eog_threshold
                self.threshlod_status = ThreshlodStatus.Normal
                self.current_count = 0
                self.sum = 0
        elif self.threshlod_status == ThreshlodStatus.Update:
            if self.current_count - self.discard_count >= 0:
                self.sum += value
                if self.current_count - self.discard_count == self.dynamic_count:
                    self.baseline = self.sum / self.current_count
                    self.blink_threshold = self.baseline + self.eog_threshold
                    self.threshlod_status = ThreshlodStatus.Normal
                    self.current_count = 0
                    self.sum = 0

    def blinkRecognition(self, value):
        if self.blink_threshold < value < (
                self.blink_threshold + 50) and self.threshlod_status == ThreshlodStatus.Normal:
            self.count += 1
            if self.count == 40:
                self.count = 0
                self.p += 1
                self.x[self.p] = 1
                dx = self.x[self.p] - self.x[self.p - 1]
                if self.p > 1023:
                    self.p = 0
                    self.x[self.p] = 0
                if dx > 0:
                    self.current_count = 0
                    self.sum = 0
                    return True
                else:
                    return False
        else:
            self.count = 0
            self.p += 1
            self.x[self.p] = 0
            dx = self.x[self.p] - self.x[self.p - 1]
            if dx < 0:
                self.threshlod_status = ThreshlodStatus.Update
            if self.p > 1023:
                self.p = 0
                self.x[self.p] = 0
            return False
