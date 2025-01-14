
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def initialize(env_file_path: str = ".env"):
    """
    Initializes the logger and environment variables.

    Params:
        env_file_path (str): The path to the environment variables file. Defaults to ".env".
    """
    logger.info("Initializing env vars...")
    if env_file_path is not None:
        env_file_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)

    if env_file_path:
        logger.info(f"Loading environment variables from: {env_file_path}")
        load_dotenv(env_file_path, verbose=True, override=True)
