from pathlib import Path
from typing import Dict

import yaml


def load_yaml(yaml_path: Path) -> Dict:
    return yaml.load(open(yaml_path), Loader=yaml.BaseLoader)
