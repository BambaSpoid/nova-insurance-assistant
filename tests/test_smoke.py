from nova_assistant import __version__
from nova_assistant.config import get_settings


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_default_settings() -> None:
    settings = get_settings()

    assert settings.app_title == "Nova Insurance Assistant"
    assert settings.environment == "development"
    assert settings.data_dir.as_posix() == "data"