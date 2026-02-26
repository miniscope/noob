class NoobError(Exception):
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


class SchedulerError(NoobError):
    """Base error in the scheduler"""


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


class EpochCompletedError(SchedulerError, ValueError):
    """
    An epoch was already completed, but some attempt was made to update it or use it.
    """


class EpochExistsError(SchedulerError, ValueError):
    """
    Epoch already exists and is active, but attempted to create it.
    """


class NotOutYetError(SchedulerError, ValueError):
    """
    Node was marked done but wasn't passed out yet!
    """


class NotAddedError(SchedulerError, ValueError):
    """
    Node was marked done but wasn't added!
    """


class AlreadyDoneError(SchedulerError, ValueError):
    """
    Node was marked done, but it was already done!
    """
