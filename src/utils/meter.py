#!/usr/bin/env python
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.values = None
        self.tensor_initialized = False

    def update(self, value):
        if not self.tensor_initialized:
            self.values = value
            self.tensor_initialized = True
        else:
            self.values = torch.cat((self.values, value), dim=0) #concatenate along batch dimension

    @property
    def std(self):
        return torch.std(self.values, dim=0, unbiased=True).item()

    @property
    def avg(self):
        return torch.mean(self.values, dim=0).item()

    @property
    def num_samples(self):
        return self.values.shape[0]


class MetricLogger(object):
    def __init__(self, delimiter="\n", print_precision="{:.2f}"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_precision = print_precision

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor), "Metric to update should be a torch tensor"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        info_order = "\nName: avg (std) num_samples\n"
        base_str = "{}: " + self.print_precision + " (" + self.print_precision + ") {}"
        for name, meter in self.meters.items():
            loss_str.append(base_str.format(name, meter.avg, meter.std, meter.num_samples))

        return info_order + self.delimiter.join(loss_str)
