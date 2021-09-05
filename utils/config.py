import yaml


class ConfigParser:
    def __init__(self, config_path: str) -> None:
        assert config_path.endswith(tuple([".yaml", ".yml"]))
        self.config_path = config_path
        self.read_config()

    def read_config(self):
        self.config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
        for k, v in self.config.items():
            setattr(self, k, v)
