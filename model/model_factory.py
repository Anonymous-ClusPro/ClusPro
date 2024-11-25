
from model.cluspro import cluspro

def get_model(config, attributes, classes, offset):
    if config.model_name == 'cluspro':
        model = cluspro(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )


    return model