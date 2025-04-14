systems = {}


def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, config_explicit_implicit, load_from_checkpoint=None):
    if load_from_checkpoint is None:
        system = systems[name](config, config_explicit_implicit)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config, global_config=config_explicit_implicit)
    return system


from . import neus_ir_avatar
from . import neus_ir_avatar_GS