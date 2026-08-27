import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration générale de l'application."""

    app_title: str
    environment: str
    data_dir: Path


def get_settings() -> Settings:
    """Charge la configuration depuis les variables d'environnement."""

    return Settings(
        app_title=os.getenv("NOVA_APP_TITLE", "Nova Insurance Assistant"),
        environment=os.getenv("NOVA_ENV", "development"),
        data_dir=Path(os.getenv("NOVA_DATA_DIR", "data")),
    )
