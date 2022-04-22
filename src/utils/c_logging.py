import json
import logging
from logging.handlers import RotatingFileHandler
import os


def log_json(logger, log_level, type_name, json):
    if logger.isEnabledFor(log_level):
        logger.log(log_level, 'JSON - %s - %s', type_name, json)


def log_item(logger, log_level, type_name, item):
    if logger.isEnabledFor(log_level):
        log_json(logger, log_level, type_name, item if isinstance(item, dict) else json.dumps(item.__dict__))


def getLogger(name):
    logger = logging.getLogger(name)
    logger.log_item = lambda log_level, type_name, item: log_item(logger, log_level, type_name, item)
    logger.log_json = lambda log_level, type_name, item: log_json(logger, log_level, type_name, item)
    return logger


def config(project, experiment_id, experiment_dir, log_level=logging.INFO, log_to_stream=True):
    log = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    file_handler = RotatingFileHandler(os.path.join(experiment_dir, '{}-{}.log'.format(project, experiment_id)))
    file_handler.setFormatter(formatter)
    
    log.addHandler(file_handler)
    if log_to_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)

    log.setLevel(log_level)
    log.info('Logging %s %s', project, experiment_id)
