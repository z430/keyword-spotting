import yaml


class KWSConfig:
    def __init__(self, config_path: str) -> None:
        assert config_path.endswith(tuple(["yaml, yml"]))
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
