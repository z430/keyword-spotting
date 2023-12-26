from pathlib import Path

from kws.data.dataloaders import SpeechCommandsDataset

ROOT_DIR = Path(__file__).parent


def main():
    dataset = SpeechCommandsDataset(
        dataset_path=(ROOT_DIR / "data" / "dataset"),
        config_path=(ROOT_DIR / "configs" / "parameters.yaml"),
    )


if __name__ == "__main__":
    main()
