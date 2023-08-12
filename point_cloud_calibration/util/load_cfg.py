import json
from abc import ABC, abstractmethod


class ConfigureLoader:
    def __init__(self, path):
        self.maxPoints = 0
        with open(path, "r") as f:
            cfg = json.load(f)

        if "mlp.json" in path:
            self._load_mlp_cfg(cfg)
        elif "pointNet.json" in path:
            self._load_pointNet_cfg(cfg)
        elif "cnn.json" in path:
            self._load_cnn_cfg(cfg)
        else:
            self.__loader__(cfg)

    @abstractmethod
    def __loader__(self, cfg): pass

    def _load_mlp_cfg(self, cfg):
        self.k = cfg["k"]
        self.model = cfg["model"]
        self.layers = cfg["layers"]
        self.batch = cfg["batch_size"]

    def _load_pointNet_cfg(self, cfg):
        self.k = cfg["k"]
        self.model = cfg["model"]
        self.batch = cfg["batch_size"]
        self.globalFeature = cfg["global_feature"]
        self.useFeatureTrans = cfg["use_feature_trans"]
        self.maxPoints = cfg["max_points"]

    def _load_cnn_cfg(self, cfg):
        self.k = cfg["k"]
        self.model = cfg["model"]
        self.batch = cfg["batch_size"]
        self.maxPoints = cfg["max_points"]
