# 通信线程，主要负责与采集的通信
import json
import socket
import struct
import threading
import time
import numpy as np

lock = threading.Lock()


class Communication(threading.Thread):
    def __init__(self, channel_num, samplerate):
        super().__init__()
        # 数据服务端口
        self.data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_server.bind(("", 7777))
        self.data_server.setblocking(False)
        self.data_server.listen(1)
        self.data_client = None
        # 接收数据
        self.recv_data = None
        # 命令服务端口
        self.command_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.command_server.bind(("", 7778))
        self.command_server.setblocking(False)
        self.command_server.listen(1)
        self.command_client = None
        # 解析配置
        self.channel_num = channel_num
        self.samplerate = samplerate
        self.time_seq = 5
        self.data_slice = 1

    def get(self, data_len):
        data = self.recv_data[data_len]
        self.recv_data = np.delete(self.recv_data, [0, 1, 2, 3, 4], axis=0)
        return data

    def get_len(self):
        return self.recv_data.shape[1]

    def data_recv(self):
        if self.data_client:
            lock.acquire()
            try:
                recv_data = self.data_client.recv(self.channel_num * 8 * self.samplerate * self.data_slice)
                if not recv_data:
                    self.data_client = None
                else:
                    recv_data = struct.unpack("{}d".format(self.channel_num * self.samplerate * self.data_slice))
                    recv_data = np.array(recv_data)
                    recv_data = np.reshape(recv_data, (self.samplerate, self.channel_num))
                    recv_data = np.expand_dims(recv_data, axis=0)
                    if self.recv_data is None:
                        self.recv_data = recv_data
                    else:
                        self.recv_data = np.concatenate([self.recv_data, recv_data], axis=0)
            except socket.error as e:
                if e.errno == 10054:
                    self.data_client = None
                if e.errno == 10053:
                    self.data_client = None
            lock.release()
        else:
            try:
                self.data_client, _ = self.data_server.accept()
                self.recv_data = None
                print("采集连接")
            except Exception as e:
                print("没有连接")

    def back_result(self, result):
        result = json.dumps(result).encode(encoding="utf8")
        self.data_client.send(result)
        print("计算结果：{}".format(result))

    def command_recv(self):
        if self.command_client:
            try:
                data = self.command_client.recv(self.channel_num * 8 * self.samplerate)
                if not data:
                    self.command_client = None
                else:
                    command = str(data, 'utf-8')
                    print(command)
            except socket.error as e:
                if e.errno == 10054:
                    self.command_client = None
                if e.errno == 10053:
                    self.command_client = None
        else:
            try:
                self.command_client, _ = self.command_server.accept()
                self.recv_data = None
                print("采集连接")
            except Exception as e:
                print("没有连接")

    def run(self):
        while (1):
            self.data_recv()
            self.command_recv()
