import yaml
import os

from .models import Configuration


def load(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified config file does not exist: {file_path}")

    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return Configuration(**config)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")


app_config = load(os.getenv("CONFIG_PATH", "./config.yaml"))
