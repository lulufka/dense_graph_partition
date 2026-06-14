import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KapoceConfig:
    executable_path: Path
    config_path: Path


def load_local_config() -> dict[str, Any]:
    """
   Loads local configuration variables from a JSON file.
   The file 'config.local.json' is expected to contain paths to external tools like the KaPoCE executable.

   Returns:
       dict[str, Any]: A dictionary containing configuration values.

   Raises:
       FileNotFoundError: If 'config.local.json' is not found in the working directory.
   """
    path = Path("config.local.json")

    if not path.exists():
        raise FileNotFoundError(
            "config.local.json not found. Please create it with your local paths."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_kapoce_config() -> KapoceConfig:
    config = load_local_config()
    return KapoceConfig(
        executable_path=Path(config["kapoce_executable"]),
        config_path=Path(config["kapoce_config"]),
    )

