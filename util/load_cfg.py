import json


class ConfigureLoader:
    def __init__(self, path):
        with open(path, "r") as f:
            cfg = json.load(f)

        if "mlp.json" in path:
            self._load_mlp_cfg(cfg)
        elif "pointNet.json" in path:
            self._load_pointNet_cfg(cfg)
        else:
            self.__loader__(cfg)


    def __loader__(self, cfg):
        """
        Implement this method if you want to use your own model
        :param cfg: Your own config file
        :return:
        """
        raise RuntimeError("You should implement this method at every child class")

    def _load_mlp_cfg(self, cfg):
        self._k = cfg["k"]
        self._layers = cfg["layers"]
        self._batch = cfg["batch_size"]

    def _load_pointNet_cfg(self, cfg):
        self._k = cfg["k"]
        self._batch = cfg["batch_size"]
        self._globalFeature = cfg["global_feature"]
        self._useFeatureTrans = cfg["use_feature_trans"]

    @property
    def k(self): return self._k
    @property
    def layers(self): return self._layers
    @property
    def batch(self): return self._batch
    @property
    def globalFeature(self): return self._globalFeature
    @property
    def useFeatureTrans(self): return self._useFeatureTrans
