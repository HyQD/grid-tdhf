import uuid


def get_io_overrides(system_config, output_dir="output/"):
    if system_config.output_dir is not None:
        output_dir = system_config.output_dir

    if system_config.load_run is not None:
        fileroot = system_config.load_run
    elif system_config.fileroot is not None:
        fileroot = system_config.fileroot
    else:
        fileroot = str(uuid.uuid4())

    overrides = {
        "fileroot": fileroot,
        "output_dir": output_dir,
    }

    return overrides
