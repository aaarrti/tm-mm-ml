import inspect
from pathlib import Path
import functools


def log_before(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


def log_after(func):
    @functools.wraps(func)
    def wrapper(*func_args, **func_kwargs):
        retval = func(*func_args, **func_kwargs)
        print('Exited ' + func.__name__ + '() with value: ' + repr(retval))
        return retval

    return wrapper


def ensure_dir(file_path):
    Path(file_path).mkdir(exist_ok=True)


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance
