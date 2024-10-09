import os
import logging
import glob

import torch

import fedora.customtypes as fT

logger = logging.getLogger("Simulator")


def make_client_checkpoint_dirs(root_dir=".", client_ids=[]):
    # os.makedirs(f"{}/client_ckpts", exist_ok=True)
    for cid in client_ids:
        os.makedirs(f"client_ckpts/{cid}")


def make_server_checkpoint_dirs(root_dir="."):
    os.makedirs("server_ckpts")


# def make_checkpoint_dirs(has_server: bool, client_ids=[]):
#     if has_server:
#         make_server_checkpoint_dirs()

#     os.makedirs("client_ckpts", exist_ok=True)

#     for cid in client_ids:
#         make_client_checkpoint_dirs(cid)

def checkpoint_dirs_exist(root_dir=".") -> bool:
    return os.path.exists(root_dir + "/server_ckpts") or os.path.exists(
        root_dir + "/client_ckpts"
    )


def find_server_checkpoint(root_dir=".") -> str:
    server_ckpts = sorted(glob.glob(root_dir + "/server_ckpts/server_ckpt_*"))
    if server_ckpts:
        logger.info(f"------ Found server checkpoint: {server_ckpts[-1]} ------")
        return server_ckpts[-1]
    else:
        # logger.debug("------------ No server checkpoint found. ------------")
        raise FileNotFoundError("No server checkpoint found")


def find_client_checkpoint(client_id: str, root_dir=".") -> str:
    client_ckpts = sorted(
        glob.glob(f"{root_dir}/client_ckpts/{client_id}/ckpt_*")
    )
    if client_ckpts:
        logger.info(
            f"------ Found client {client_id} checkpoint: {client_ckpts[-1]} ------"
        )
        return client_ckpts[-1]
    else:
        # logger.debug(f"------------ No client {client_id} checkpoint found. Starting afresh ------------")
        raise FileNotFoundError(f"No client {client_id} checkpoint found")


    
def parameter_average_aggregation(client_params: fT.ClientParams_t) -> fT.ActorParams_t:
    """Naive averaging of client parameters"""
    server_params = {}
    client_ids = list(client_params.keys())
    param_keys = client_params[client_ids[0]].keys()
    for key in param_keys:
        server_params[key] = torch.stack(
            [client[key].data for client in client_params.values()]
        ).mean(dim=0)
    return server_params
