import hydra
from omegaconf import OmegaConf
import logging
import sys
import os
import cProfile, pstats
from hydra.core.hydra_config import HydraConfig

from fedora.simulator.simulator import Simulator
from fedora.config.masterconf import Config, register_configs

# Icecream debugger
from icecream import install, ic

install()
ic.configureOutput(includeContext=True)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
# @hydra.main(version_base=None, config_path="conf", config_name="flowerconfig")
def run_fedora(cfg: Config):

    if '--experimental-rerun' in sys.argv:
        os.chdir(HydraConfig.get().runtime.output_dir)
    
    cfg_obj: Config = OmegaConf.to_object(cfg)  # type: ignore

    # logger.info(f"[Hydra MODE] : {OmegaConf.to_yaml(HydraConfig.get())}")
    
    logger.info((OmegaConf.to_yaml(cfg_obj)))
    if cfg_obj.mode != "debug":
        input("Review Config. Press Enter to continue...")
    # logger.debug(cfg_obj.split.__dict__)
    # logger.debug(cfg_obj.client.cfg.__dict__)

    sim = Simulator(cfg_obj)
    sim.run_simulation()
    sim.close()


if __name__ == "__main__":
    register_configs()
    # cProfile.run('run_fedora()', 'feduciary_stats', pstats.SortKey.CUMULATIVE)
    run_fedora()
