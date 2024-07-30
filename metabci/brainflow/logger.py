# -*- coding: utf-8 -*-
# License: MIT License
"""
Logging system.

"""
import logging
from logging.handlers import RotatingFileHandler


class LevelFormatter(logging.Filter):
    def __init__(self, level, fmt):
        super().__init__()
        self.level = level
        self.fmt = fmt

    def filter(self, record):
        if record.levelno == self.level:
            # 使用已有的 formatter 来格式化记录
            record.msg = self.fmt.format(record)
        return True


def set_format(handler):
    # 根据信息的等级不同创建不同的日志格式
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    warning_formatter = logging.Formatter('%(levelname)s - %(message)s')
    error_formatter = logging.Formatter(
        '%(asctime)s - %(filename)s -%(funcName)s - %(lineno)d - %(levelname)s - %(message)s')

    info_filter = LevelFormatter(logging.INFO, info_formatter)
    warning_filter = LevelFormatter(logging.WARNING, warning_formatter)
    error_filter = LevelFormatter(logging.ERROR, error_formatter)

    handler.addFilter(info_filter)
    handler.addFilter(warning_filter)
    handler.addFilter(error_filter)


def get_logger(log_name, out_in_console=False, rotating_log=False, rotating_log_size=0, rotating_log_count=0):
    """get system logger.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        Nonw
    Parameters
    ----------
    rotating_log_count: int,
        Maximum log count
    rotating_log_size: int,
        Maximum log size, kb
    rotating_log: bool,
        Choose whether to roll back the log
    out_in_console: bool,
        Decide whether to display it on the console
    log_name: str,
        Name of logger.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(log_name + ".txt", encoding="utf-8")
    handler.setLevel(level=logging.INFO)
    set_format(handler)
    logger.addHandler(handler)
    if out_in_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        set_format(console)
        logger.addHandler(console)
    if rotating_log:
        rHandler = RotatingFileHandler(log_name + ".txt", maxBytes=1024 * rotating_log_size,
                                       backupCount=rotating_log_count)
        rHandler.setLevel(logging.INFO)
        set_format(rHandler)
        logger.addHandler(rHandler)
    return logger


def disable_log():
    """disable system logger.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        Nonw
    """
    logging.disable(logging.INFO)


def apply_log():
    # 重新启用日志
    logging.disable(logging.NOTSET)
