import time

from htrflow_core.logging.logger import CustomLogger


logger = CustomLogger("ProfileLogger", verbose=3)


def profile_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # Use the custom logger for logging
        logger.logger.debug(f"Function: {func.__name__}, Execution Time: {execution_time} seconds")
        return result

    return wrapper


if __name__ == "__main__":

    @profile_performance
    def example_function(x):
        result = 0
        for i in range(x):
            result += i
        return result

    example_function(100)
