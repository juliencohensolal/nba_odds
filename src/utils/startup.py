import numpy as np
import os
import random

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
