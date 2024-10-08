from collections import defaultdict
import pickle
from importlib import import_module
from .metricszoo import BaseMetric
import fedora.customtypes as fT
from fedora.utils import get_time
from fedora.config.commonconf import MetricConfig
from dataclasses import dataclass, field
from typing import Optional
import os
from copy import deepcopy
import json

# TODO: Consider merging with Result Manager Later
##################
# Metric manager #
##################


def _result_to_mea_dict(result: fT.Result):
    out = {}
    for metric, value in result.metrics.items():
        out[metric] = {result.event: {result.actor: value}}
    for meta, value in result.metadata.items():
        out[meta] = {result.event: {result.actor: value}}
    out["size"] = {result.event: {result.actor: result.size}}
    return out


def _save_pickle(obj, actor, out_prefix="", root="temp"):
    files = [
        filename for filename in os.listdir(root) if filename.startswith(f"{actor}")
    ]

    if files:
        files.sort(reverse=True)
        # print(files[0].removeprefix(f'{actor}_').removesuffix('.pickle'))
        last_num = int(files[0].removeprefix(f"{actor}_").removesuffix(".pickle")) + 1
    else:
        last_num = 0

    filename = f"{root}/{out_prefix}{actor}_{last_num}.pickle"
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)


class MetricManager:
    """Lightweight class to compute metrics and log to pickle files."""

    def __init__(self, cfg: MetricConfig, _round: int, actor: str):
        self.cfg = cfg
        eval_metrics = cfg.eval_metrics
        self.metric_funcs: dict[str, BaseMetric] = {
            name: import_module(f".metricszoo", package=__package__).__dict__[
                name.title()
            ]()
            for name in eval_metrics
        }
        self.figures = defaultdict(int)
        self._result = fT.Result(_round=_round, actor=actor)
        self._round = _round
        self._actor = actor
        self._log_to_file = cfg.log_to_file

        # self._pmea_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._pmea_dict = {}

        # Move it to a central directory creator to reduce overheads
        if cfg.log_to_file:
            # Hack to make the files visible in the project workspace
            self._temp_dir = f"{cfg.cwd}/temp/{actor}"  # Possibly breaks on resume
            os.makedirs(self._temp_dir, exist_ok=True)

    def track(self, loss, pred, true):
        # update running loss
        self.figures["loss"] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, epoch) -> fT.Result:
        # aggregate
        # print(file_prefix := f'{self.cfg.file_prefix}{self._actor}_{self._round}')
        avg_metrics = {
            name: module.summarize(
                out_prefix=f"{self.cfg.cwd}/{self.cfg.file_prefix}{self._actor}_{self._round}"
            ) # Possibly breaks on resume
            for name, module in self.metric_funcs.items()
        }

        avg_metrics["loss"] = self.figures["loss"] / total_len

        self._result.metrics = avg_metrics
        self._result.metadata["epoch"] = epoch
        self._result.size = total_len
        self._result._round = self._round

        if self._log_to_file:
            self._pmea_dict[self._result.phase] = _result_to_mea_dict(self._result)

        self.figures = defaultdict(int)

        # _save_pickle(self._pmea_dict, self._actor, root=self._temp_dir)

        return self._result

    def flush(self):
        self.figures = defaultdict(int)
        self._result = fT.Result(_round=self._round, actor=self._actor)

    def _add_metric(self, metric, event, phase, actor, value):
        # Metric addition of what metric, what contxt, when and who and what value
        self._pmea_dict[phase] = {metric: {event: {actor: value}}}

    def log_general_metric(
        self, metric_val, metric_name: str, phase: str, event: str = ""
    ):
        if isinstance(metric_val, dict):
            for key, val in metric_val.items():
                self.log_general_metric(val, f"{metric_name}/{key}", phase, event)
        elif isinstance(metric_val, (float, int)):
            self._add_metric(metric_name, event, phase, self._actor, metric_val)
        else:
            err_str = f"Metric logging for {metric_name} of type: {type(metric_val)} is not supported"
            raise TypeError(err_str)

    def json_dump(self, metric_val, metric_name: str, phase: str, event: str = ""):
        if not isinstance(metric_val, (float, int, dict, list)):
            err_str = f"Metric logging for {metric_name} of type: {type(metric_val)} is not supported"
            raise TypeError(err_str)
        else:
            os.makedirs(f"debug/{self._actor}/r_{self._round}", exist_ok=True)
            with open(
                f"debug/{self._actor}/r_{self._round}/{metric_name}_{event}_{phase}.json",
                "w",
            ) as f:
                json.dump(metric_val, f, indent=4)

    def __del__(self):
        if self._log_to_file:
            # with get_time():
            _save_pickle(deepcopy(self._pmea_dict), self._actor, root=self._temp_dir)
