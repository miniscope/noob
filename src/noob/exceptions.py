class NoobError(Exception):
    """Base exception type"""


# -----------------------------------------------------
# Top-level error categories
# use these as a mixin with another base exception type
# -----------------------------------------------------


class ConfigError(NoobError):
    """Base config error type"""


class StoreError(NoobError):
    """Base store error type"""


class TubeError(NoobError):
    """Base tube error type"""


class RunnerError(NoobError):
    """Base runner error type"""


class InputError(NoobError):
    """Error with tube input"""


# --------------------------------------------------
# Actual error types you should use
# --------------------------------------------------


class ConfigMismatchError(ConfigError, ValueError):
    """
    Mismatch between the fields in some config model and the fields in the model it is configuring
    """


class AlreadyRunningError(RunnerError, RuntimeError):
    """
    A tube is already running!
    """


class InputMissingError(InputError, ValueError):
    """
    A requested input was not provided in the given scope
    """
