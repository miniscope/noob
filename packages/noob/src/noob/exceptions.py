from noob_core.exceptions import *  # noqa: F403
from noob_core.exceptions import NoobCoreError


class NoobError(NoobCoreError):
    """Base exception type"""


class NoobWarning(UserWarning):
    """Base warning type"""


# -----------------------------------------------------
# Top-level error categories
# use these as a mixin with another base exception type
# -----------------------------------------------------


class ConfigError(NoobError):
    """Base config error type"""


class ConfigWarning(NoobWarning):
    """Base config warning type"""


class StoreError(NoobError):
    """Base store error type"""


class TubeError(NoobError):
    """Base tube error type"""


class RunnerError(NoobError):
    """Base runner error type"""


class InputError(NoobError):
    """Error with tube input"""


class InputWarning(NoobWarning):
    """Warning with tube input"""


# --------------------------------------------------
# Actual error types you should use
# --------------------------------------------------


class ConfigMismatchError(ConfigError, ValueError):
    """
    Mismatch between the fields in some config model and the fields in the model it is configuring
    """


class EntrypointImportWarning(ConfigWarning, ImportWarning):
    """Some problem with a configuration entypoint, usually when importing"""


class AlreadyRunningError(RunnerError, RuntimeError):
    """
    A tube is already running!
    """


class InputMissingError(InputError, ValueError):
    """
    A requested input was not provided in the given scope
    """


class ExtraInputError(InputError, ValueError):
    """
    Extra input was provided in some scope where it was not specified
    """


class ExtraInputWarning(InputWarning, RuntimeWarning):
    """
    Extra input was provided in some scope where it was not specified,
    but it was ignorable
    """


class TerminateTaskGroup(NoobError):
    """
    https://docs.python.org/3/library/asyncio-task.html#terminating-a-task-group
    """
