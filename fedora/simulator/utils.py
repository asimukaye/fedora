import os
import logging
import glob

import torch

import fedora.customtypes as fT
logger = logging.getLogger('Simulator')


def make_client_checkpoint_dirs(client_id: str):
    os.makedirs("client_ckpts", exist_ok=True)
    os.makedirs(f"client_ckpts/{client_id}", exist_ok=True)

def make_server_checkpoint_dirs():
    os.makedirs("server_ckpts", exist_ok=True)
    
def make_checkpoint_dirs(has_server: bool, client_ids=[]):
    if has_server:
        make_server_checkpoint_dirs()

    os.makedirs("client_ckpts", exist_ok=True)

    for cid in client_ids:
        make_client_checkpoint_dirs(cid)



def find_checkpoint() -> tuple[str, dict]:
    server_ckpts = sorted(glob.glob("server_ckpts/server_ckpt_*"))
    client_ckpts = {}

    for dir in os.listdir("client_ckpts/"):
        files = sorted(os.listdir(f"client_ckpts/{dir}"))
        if files:
            client_ckpts[dir] = f"client_ckpts/{dir}/{files[-1]}"
            logger.info(
                f"------ Found client {dir} checkpoint: {client_ckpts[dir]} ------"
            )

    if server_ckpts or client_ckpts:
        if server_ckpts:
            logger.info(f"------ Found server checkpoint: {server_ckpts[-1]} ------")
            return server_ckpts[-1], client_ckpts
        else:
            return "", client_ckpts
    else:
        logger.debug("------------ No checkpoints found. Starting afresh ------------")
        return "", {}


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

