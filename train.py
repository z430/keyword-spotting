from pathlib import Path

from loguru import logger

from kws.datasets.speech_commands import SpeechCommandsDataset

ROOT_DIR = Path(__file__).parent


def main():
    dataset = SpeechCommandsDataset(
        ROOT_DIR / "configs" / "speech_commands_parameters.yaml"
    )
    logger.info(dataset.parameters)


if __name__ == "__main__":
    main()
