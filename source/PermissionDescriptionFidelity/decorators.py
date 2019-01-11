import functools
import time

from log import logger

def logging(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  
        signature = ", ".join(args_repr + kwargs_repr)           
        logger.info(f"Calling {func.__name__}({signature})")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        logger.info(f"{func.__name__!r} returned {value!r}")
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_debug