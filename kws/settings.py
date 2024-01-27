from pathlib import Path

ROOT_PATH = Path(__file__).parent
CONFIG_SPEECHCOMMANDS_PATH = (
    ROOT_PATH.parent / "configs" / "speech_commands_parameters.yaml"
)
CONFIG_SPEECHCOMMANDS_DATASET_PATH = ROOT_PATH.parent / "data" / "speech_commands"
