import sys
import time
import warnings


class IncompatibleVersionError(Exception):
    """Raised when the version of runn used to create the model is not compatible with the current version."""

    pass


class NotSupportedError(Exception):
    """Raised when a not supported operation for a given model is called."""

    pass


class WarningManager:
    """Singleton class to manage warnings."""

    _instance = None
    _show_warnings = True

    def __new__(cls, *args, **kwargs) -> "WarningManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.show_warnings = True
        return cls._instance

    def warn(self, message: str) -> None:
        """Show a warning message.

        Args:
            message: The warning message.
        """
        if not isinstance(message, str):
            raise ValueError("The message should be a string.")
        if self.show_warnings:
            warnings.warn(message)

    def set_show_warnings(self, show_warnings: bool) -> None:
        """Set whether to show warnings or not.

        Args:
            show_warnings: Whether to show warnings or not.
        """
        self.show_warnings = show_warnings


class ProgressBar:
    """A simple progress bar.

    Args:
        total: The total number of steps.
        bar_length: The length of the bar.
        filled_mark: The character to fill the bar.
        suffix: The suffix to be shown at the end of the bar.
    """

    def __init__(self, total, bar_length=25, filled_mark="=", suffix=""):
        self.total = total
        self.bar_length = bar_length
        self.filled_mark = filled_mark
        self.suffix = suffix
        self.start_time = None

    def update(self, count_value, value_dict={}):
        """Update the progress bar.

        Args:
            count_value: The current value.
            value_dict: A dictionary of values to be shown at the end of the bar.
        """
        if self.start_time is None:
            self.start_time = time.time()
        if count_value > self.total:
            count_value = self.total
        filled_up_Length = int(round(self.bar_length * count_value / float(self.total)))
        percentage = int(round(100.0 * count_value / float(self.total)))
        bar = self.filled_mark * filled_up_Length + " " * (self.bar_length - filled_up_Length)
        time_sufix = " - Elapsed: {:d}s".format(int(round(time.time() - self.start_time)))
        value_sufix = ""
        if len(value_dict) > 0:
            for key, value in value_dict.items():
                value_sufix += " - {}: {}".format(key, value)
        n_digits = len(str(self.total))
        sys.stdout.write(
            "\r {:{}}/{} ({:3d}%) [{}] {}{}".format(
                count_value, n_digits, self.total, percentage, bar, time_sufix, value_sufix
            )
        )
        sys.stdout.flush()
        if count_value >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.start_time = None
