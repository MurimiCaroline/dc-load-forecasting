import pytest

from dcforecast import config


@pytest.fixture(scope="session")
def built_pipeline():
    """Build every layer and the models once per test session if they are missing.

    The raw dataset is not committed to the repository, so on a fresh clone (or in CI)
    without data/raw/ the API tests are skipped; the validation and leakage tests still run.
    """
    if not (config.MODELS_DIR / "metadata.json").exists():
        if not config.RAW_FILE.exists():
            pytest.skip(f"Raw data not found at {config.RAW_FILE}; download it to run the API tests.")
        from dcforecast import pipeline
        pipeline.main(["--stage", "all"])
    return True
