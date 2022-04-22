import time

from execution.preprocessor import Preprocessor
from execution.oddsmaker import Oddsmaker
from utils import c_logging, config, startup


LOG = c_logging.getLogger(__name__)


if __name__ == "__main__":
    # Load config
    conf = config.load_config("conf/conf.yml")

    # Init logging
    project = "NBA_Challenge"
    experiment_id = int(time.time())
    experiment_dir = "./logs/" + project + "_" + str(experiment_id) + "/"
    c_logging.config(
        project=project,
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        log_level=conf.log_level, 
        log_to_stream=conf.log_to_stream)

    # Setup everything
    startup.seed_everything(conf.seed)
    config.save_config("conf", "conf.yml", experiment_dir)

    LOG.info("AAAND NOOOW... THE STARTING LINEUP... FOR THE WORLD CHAMPIONS.... THE NEW YORK KNICKS!")

    # Start execution in requested mode
    if conf.mode == "prepare":
        LOG.info("TASK: Prepare data")
        preprocessor = Preprocessor(conf)
        preprocessor.preprocess()
    elif conf.mode == "train":
        LOG.info("TASK: Train model which will later be used to predict a season's odds")
        oddsmaker = Oddsmaker(conf, experiment_id)
        oddsmaker.train_model()
    elif conf.mode == "before_regular":
        LOG.info("TASK: Give championship odds before beginning of 2018/2019 regular season")
        oddsmaker = Oddsmaker(conf, experiment_id)
        oddsmaker.before_regular_predict()
    elif conf.mode == "before_playoffs":
        LOG.info("TASK: Give championship odds before beginning of 2018/2019 playoffs")
        oddsmaker = Oddsmaker(conf, experiment_id)
        oddsmaker.before_playoffs_predict()
    else:
        raise NotImplementedError("mode " + conf.mode + " not implemented")
