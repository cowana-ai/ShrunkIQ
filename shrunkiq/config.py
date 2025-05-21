from hydra import compose, initialize


def load_config():
    with initialize(version_base="1.3.2", config_path="."):
        cfg = compose(config_name="config")
        return cfg


cfg = load_config()
