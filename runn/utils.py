import warnings


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
