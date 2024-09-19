# -*- coding: utf-8 -*-
import os
from loguru import logger

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def __setting__():
    logger.add(
        os.path.join(base_dir, "logs/ERROR/{time:YYYY-MM-DD}-error.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        filter=lambda x: True if x["level"].name == "ERROR" else False,
        rotation="00:00", retention=7, level='ERROR', encoding='utf-8'
    )
    logger.add(
        os.path.join(base_dir, "logs/SUCCESS/{time:YYYY-MM-DD}-success.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        filter=lambda x: True if x["level"].name == "SUCCESS" else False,
        rotation="00:00", retention=7, level='SUCCESS', encoding='utf-8',
    )
    logger.add(
        os.path.join(base_dir, "logs/INFO/{time:YYYY-MM-DD}-info.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        rotation="00:00", retention=7, level='INFO', encoding='utf-8'
    )

    return logger


class GetLogging:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = __setting__()
        return cls._instance

    def get_instance(self):
        return self.logger

def get_logger():
    return GetLogging().get_instance()
