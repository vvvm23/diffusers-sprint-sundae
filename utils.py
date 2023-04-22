# https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace
from types import SimpleNamespace


def dict_to_namespace(d):
    x = SimpleNamespace()
    _ = [
        setattr(x, k, dict_to_namespace(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


def infinite_loader(loader):
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)
            yield next(it)
