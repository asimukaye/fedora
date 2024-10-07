import logging
from copy import deepcopy
import time

from torch.nn import Module
from torch.utils.data import Dataset, ConcatDataset, DataLoader
# from hydra.utils import instantiate

from fedora.config.masterconf import Config
from fedora.results.resultmanager import ResultManager
from fedora.results.metricmanager import MetricManager
from fedora.client.abcclient import simple_evaluator, simple_trainer

import fedora.customtypes as fT

logger = logging.getLogger(__name__)



def run_centralized_simulation(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Dataset,
    model: Module,
):

    #TODO: test and include this later
    # make_checkpoint_dirs(has_server=False, client_ids=['centralized'])

    pooled_train_set = ConcatDataset([train_set for train_set, _ in client_datasets])
    pooled_test_set = ConcatDataset([test_set for _, test_set in client_datasets])

    # cfg.train_cfg.loss_fn = instantiate(cfg.train_cfg.loss_fn)
    # cfg.train_cfg.optimizer = instantiate(cfg.train_cfg.optimizer)

    train_loader = DataLoader(
        dataset=pooled_train_set, batch_size=cfg.train_cfg.batch_size, shuffle=True
    )

    test_loader = DataLoader(
        dataset=server_dataset, batch_size=cfg.train_cfg.eval_batch_size, shuffle=False
    )

    result_manager = ResultManager(cfg.result, logger=logger)
    metric_manager = MetricManager(cfg.train_cfg.metric_cfg, 0, actor="simulator")

    # keeping the same amount of training duration as federated:
    total_epochs = cfg.train_cfg.epochs*cfg.simulator.num_rounds

    for curr_round in range(total_epochs):
        logger.info(f"-------- Round: {curr_round} --------\n")

        loop_start = time.time()
        train_result = simple_trainer(
            model, train_loader, cfg.train_cfg, metric_manager, curr_round
        )

        result_manager.log_general_result(
            train_result, "post_train", "sim", "central_train"
        )

        # with get_time():
        params = dict(model.named_parameters())
        # model.
        # for param_key, param in params.items():
        #     ic(param_key, param.size())
        #     ic(param_key, param.type())

        # for param_key, param in model.named_parameters():
        #     ic(param_key, param.size())
        #     ic(param_key, param.type())
        result_manager.log_parameters(params, "post_train", "sim", verbose=False)

        eval_result = simple_evaluator(
            model, test_loader, cfg.train_cfg, metric_manager, curr_round
        )
        result_manager.log_general_result(
            eval_result, "post_eval", "sim", "central_eval"
        )

        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------"
        )
    final_result = result_manager.finalize()
    return final_result


