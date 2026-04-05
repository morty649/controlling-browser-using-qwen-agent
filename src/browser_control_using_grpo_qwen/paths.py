from pathlib import Path

def get_path_to_configs() -> str:
    path = str(Path(__file__).parent.parent.parent/"configuration_files")
    
    # creating the path if it does not exists
    Path(path).mkdir(parents=True,exist_ok=True)

    return path

def get_path_to_media() -> str:
    path = str(Path(__file__).parent.parent.parent / "media")

    # create path if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    return path

def get_path_model_checkpoints(experiment_name:str) -> str:
    '''Returns path to a cached dataset in a model volume'''

    path = Path("/model_checkpoints") / experiment_name.replace("/","--")

    if not path.exists:
        path.mkdir(parents=True,exist_ok=True)

    return str(path)

