from abc import ABC, abstractmethod

import torch
import numpy as np
import warnings
import os
from torch import Tensor
import json
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,\
    average_precision_score, f1_score, precision_score, recall_score,\
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,\
            r2_score, d2_pinball_score, top_k_accuracy_score




class BaseMetric(ABC):
    '''wrapper for computing metrics over a list of values'''
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred:Tensor, true:Tensor):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    @abstractmethod
    def summarize(self, out_prefix:str =''):
        pass



class Acc1(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self, out_prefix: str = ''):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []

        # files = [filename for filename in os.listdir('temp') if filename.startswith(f'{out_prefix}answers')]

        # if files:
        #     files.sort(reverse=True)
        #     last_num = int(files[0].removeprefix(f'{out_prefix}answers_').removesuffix('.json')) + 1
        # else:
        #     last_num = 0

        # with open(f'{out_prefix}answers_{last_num}.json', 'w') as answers_json:
        #     json.dump(answers.tolist(), answers_json)
        # # print(f'scores: {scores.shape}')
        # print(f'answers: {answers.shape}')
        # with open(f'{out_prefix}scores.json', 'w') as scores_json:
        #     json.dump(scores.tolist(), scores_json)
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
            # files = [filename for filename in os.listdir('temp') if filename.startswith(f'{out_prefix}labels')]

            # if files:
            #     files.sort(reverse=True)
            #     last_num_2 = int(files[0].removeprefix(f'{out_prefix}labels_').removesuffix('.json')) + 1
            # else:
            #     last_num_2 = 0
            # with open(f'{out_prefix}labels_{last_num_2}.json', 'w') as labels_json:
            #     json.dump(labels.tolist(), labels_json)
            # print(f'labels: {labels.shape}')
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return accuracy_score(answers, labels)


class Acc5(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        num_classes = scores.shape[-1]
        return top_k_accuracy_score(answers, scores, k=5, labels=np.arange(num_classes))


class F1(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)


class Mse(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []        
        return mean_squared_error(answers, scores)
