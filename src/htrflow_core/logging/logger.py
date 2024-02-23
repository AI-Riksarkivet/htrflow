import logging


class CustomLogger:
    def __init__(self, name, verbose=0):
        self.logger = logging.getLogger(name)
        self.set_verbose(verbose)

        self.logger.propagate = False

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def set_verbose(self, verbose):
        if verbose == 1:
            self.logger.setLevel(logging.WARNING)
        elif verbose == 2:
            self.logger.setLevel(logging.INFO)
        elif verbose >= 3:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
