# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mean_class_accuracy,
                                 mmit_mean_average_precision, top_k_accuracy)
from mmaction.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class AccMetricWeatherClassification(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'mean_class_accuracy', ),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'mean_class_accuracy', 'mmit_mean_average_precision', 'mean_average_precision'
            ]
        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_scores'] # 3*3
            label = data_sample['gt_labels']  # 3
            for item_name, score in pred.items():
                pred[item_name] = score.cpu().numpy() #pred is size 3*3
            pred["rain"] = pred['item'][0]
            pred["fog"] = pred['item'][1]
            pred["snow"] = pred['item'][2]
            _ = pred.pop("item")
            result['pred'] = pred
            if label['item'].size(0) == 1:
                # single-label
                result['label'] = label['item'].item()
            else:
                # multi-label
                label = label['item'].cpu().numpy()
                result['label'] = {"rain":label[0], "fog":label[1], "snow":label[2]}
            self.results.append(result)  #self.results = List[dict{'pred': {'rain': [], 'fog': [], 'snow': []}, 'label': {'rain': 1, 'fog': 2, 'snow': 1}}]

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch. List[dict]

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = dict()
        for item_name in results[0]['pred'].keys(): # keys = ['rain', 'fog', 'snow'] 
            preds = [x['pred'][item_name] for x in results]
            labels = [x['label'][item_name] for x in results]
            eval_result = self.calculate(preds, labels) # will be seprerate for rain, fog and snow
            eval_results.update({f'{item_name}_{k}': v for k, v in eval_result.items()}) #keys = {weather name}_{metric_name}: value
        return eval_results

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}'] = acc

            if metric == 'mean_class_accuracy':
                mean1 = mean_class_accuracy(preds, labels)
                eval_results['mean1'] = mean1

            if metric in [
                    'mean_average_precision',
                    'mmit_mean_average_precision',
            ]:
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(preds, labels)
                    eval_results['mean_average_precision'] = mAP

                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(preds, labels)
                    eval_results['mmit_mean_average_precision'] = mAP

        return eval_results

