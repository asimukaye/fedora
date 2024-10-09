import logging
from copy import deepcopy
import time
from functools import partial
from collections import defaultdict
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader, Subset
# from hydra.utils import instantiate

from fedora.config.masterconf import Config, ClientSchema, get_client_partial
from fedora.results.resultmanager import ResultManager
from fedora.results.metricmanager import MetricManager
from fedora.client.abcclient import simple_evaluator
from fedora.simulator.utils import find_client_checkpoint, find_server_checkpoint, make_client_checkpoint_dirs, make_server_checkpoint_dirs, parameter_average_aggregation
from fedora.client.baseclient import BaseFlowerClient
from fedora.server.baseserver import BaseFlowerServer
from fedora.utils import generate_client_ids, log_tqdm

import fedora.customtypes as fT

logger = logging.getLogger(__name__)


def create_clients(
    all_client_ids, client_datasets, model_instance, client_partial: partial[BaseFlowerClient]
) -> dict[str, BaseFlowerClient]:

    clients = {}
    for cid, datasets in log_tqdm(
        zip(all_client_ids, client_datasets), logger=logger, desc=f"creating clients "
    ):
        # client_id = f'{idx:04}' # potential to convert to a unique hash
        client_obj: BaseFlowerClient = client_partial(
        client_id=cid, dataset=datasets, model=deepcopy(model_instance)
        )
        # client_obj = _create_client(cid, datasets, model_instance, client_cfg)
        clients[cid] = client_obj
    return clients


def run_federated_simulation(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Dataset,
    model_instance: Module,
):
    """Runs the simulation in federated mode

    Args:
        cfg (Config): The configuration object containing simulation settings.
        client_datasets (fT.ClientDatasets_t): A dictionary of client datasets.
        server_dataset (Dataset): The dataset used by the server.
        model_instance (Module): The instance of the model used in the simulation.

    Returns:
        final_result: The final result of the simulation.
    """

    all_client_ids = generate_client_ids(cfg.simulator.num_clients)

    clients: dict[str, BaseFlowerClient] = dict()

    result_manager = ResultManager(cfg.result, logger=logger)

    strategy = cfg.strategy_partial(model=model_instance, res_man=result_manager)

    #  Server gets the test set
    # server_dataset = test_set
    # Clients get the splits of the train set with an inbuilt test set
    # client_datasets =  get_client_datasets(cfg.dataset.split_conf, train_set, test_set, match_train_distribution=False)

    # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    clients = create_clients(
        all_client_ids, client_datasets, model_instance, cfg.client_partial
    )

    # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary

    server: BaseFlowerServer = cfg.server_partial(
        model=model_instance,
        strategy=strategy,
        dataset=server_dataset,
        clients=clients,
        result_manager=result_manager,
    )

    if cfg.resumed:
        server_ckpt = find_server_checkpoint()
        client_ckpts = {cid: find_client_checkpoint(cid) for cid in all_client_ids}
        server.load_checkpoint(server_ckpt)
        _round = server._round
        for cid, client in clients.items():
            client.load_checkpoint(client_ckpts[cid])
    else:
        make_server_checkpoint_dirs()
        make_client_checkpoint_dirs(client_ids=all_client_ids)
        _round = 0
    # clients = _create_clients(client_datasets)
    # server.initialize(clients, )

    for curr_round in range(_round, cfg.simulator.num_rounds):
        logger.info(f"-------- Round: {curr_round} --------\n")
        # wandb.log({'round': curr_round})
        loop_start = time.time()
        _round = curr_round
        ## update round indicator
        server._round = curr_round

        ## update after sampling clients randomly
        # TODO: Add availability logic

        update_ids = server.update(all_client_ids)

        ## evaluate on clients not sampled (for measuring generalization performance)
        if curr_round % cfg.server.cfg.eval_every == 0:
            # Can have specific evaluations later
            eval_ids = all_client_ids
            server.server_eval()
            eval_ids = server.local_eval(all_client_ids)

        if curr_round % cfg.simulator.checkpoint_every == 0:
            server.save_checkpoint()
            for client in clients.values():
                client.save_checkpoint()

        # This is weird, needs some rearch
        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------"
        )

    final_result = result_manager.finalize()
    return final_result


def run_standalone_simulation(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Dataset,
    model: Module,
):

    central_model = deepcopy(model)
    clients: dict[str, BaseFlowerClient] = defaultdict()
    # Clients get the splits of the train set with an inbuilt test set
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)



    test_loader = DataLoader(
        dataset=server_dataset, batch_size=cfg.train_cfg.eval_batch_size, shuffle=False
    )
    # client_datasets = get_client_datasets(cfg.dataset.split_conf, train_set, test_set, match_train_distribution=False)

    result_manager = ResultManager(cfg.result, logger=logger)
    metric_manager = MetricManager(cfg.train_cfg.metric_cfg, 0, actor="simulator")

    base_client_cfg = ClientSchema(
        name="BaseFlowerClient",
        cfg=cfg.client.cfg,
        train_cfg=cfg.train_cfg,
    )
    
    client_partial = get_client_partial(base_client_cfg)
    clients = create_clients(all_client_ids, client_datasets, model, client_partial)

    if cfg.resumed:
        client_ckpts = {cid: find_client_checkpoint(cid) for cid in all_client_ids}
        for cid, client in clients.items():
            client.load_checkpoint(client_ckpts[cid])
        _round = client._round
        result_manager._round = _round
    else:
        make_client_checkpoint_dirs(client_ids=all_client_ids)
        _round = 0

    # central_train_cfg = instantiate(cfg.train_cfg)
    # central_train_cfg = partial_inig(cfg.train_cfg)

    client_params = {cid: client.model.state_dict() for cid, client in clients.items()}

    for curr_round in range(_round, cfg.simulator.num_rounds):
        train_result = {}
        eval_result = {}

        for cid, client in clients.items():
            logger.info(f"-------- Client: {cid} --------\n")
            client._round = curr_round
            client_train_ins = fT.ClientIns(
                params=client_params[cid],
                metadata={},
                request=fT.RequestType.TRAIN,
                _round=curr_round,
            )
            outcome = client.download(client_train_ins)
            client_train_res = client.upload(fT.RequestType.TRAIN)
            client_params[cid] = client_train_res.params
            train_result[cid] = client_train_res.result

            eval_ins = fT.ClientIns(
                params=client_params[cid],
                metadata={},
                request=fT.RequestType.EVAL,
                _round=curr_round,
            )

            outcome = client.download(eval_ins)
            eval_result[cid] = client.upload(fT.RequestType.EVAL).result
            if curr_round % cfg.simulator.checkpoint_every == 0:
                client.save_checkpoint(epoch=curr_round)

        # Sample a
        aggregate_params = parameter_average_aggregation(client_params)
        central_model.load_state_dict(aggregate_params)
        central_eval = simple_evaluator(
            central_model, test_loader, cfg.train_cfg, metric_manager, curr_round
        )

        result_manager.log_general_result(
            central_eval, phase="post_train", actor="sim", event="central_eval"
        )


        result_manager.log_clients_result(
            train_result, phase="post_train", event="local_train"
        )
        result_manager.log_clients_result(
            eval_result, phase="post_train", event="local_eval"
        )

        if curr_round % cfg.simulator.checkpoint_every == 0:
           # FIXME: Add step count and round logic to the checkpointing system
           torch.save(central_model.state_dict(), f"central_model_{curr_round}.pth")
        
        result_manager.flush_and_update_round(curr_round)

    final_result = result_manager.finalize()
    return final_result


# FIXME: complete this function to make single client simulation work
def run_single_client(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Subset,
    model: Module,
):
    # Reusing clients dictionary to repurpose existing code
    clients: dict[str, BaseFlowerClient] = defaultdict()

    make_client_checkpoint_dirs(client_ids=["single_client"])

    # Modify the dataset here:
 
    logger.info(f"[DATA_SPLIT] Simulated dataset split : `{cfg.split.name}`")

    result_manager = ResultManager(cfg.result, logger=logger)

    for curr_round in range(cfg.simulator.num_rounds):
        logger.info(f"-------- Round: {curr_round} --------\n")
        loop_start = time.time()
        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------"
        )
    final_result = result_manager.finalize()
    return final_result
