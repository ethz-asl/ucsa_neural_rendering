import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if not "ENV_WORKSTATION_NAME" in os.environ:
    os.environ["ENV_WORKSTATION_NAME"] = "env"
