__all__ = [
    "NoobCoreError",
    "SchedulerError",
    "EpochCompletedError",
    "EpochExistsError",
    "NotAddedError",
    "AlreadyDoneError",
]


class NoobCoreError(Exception): ...


class SchedulerError(NoobCoreError):
    """Base error in the scheduler"""


class EpochCompletedError(SchedulerError, ValueError):
    """
    An epoch was already completed, but some attempt was made to update it or use it.
    """


class EpochExistsError(SchedulerError, ValueError):
    """
    Epoch already exists and is active, but attempted to create it.
    """


class NotAddedError(SchedulerError, ValueError):
    """
    Node was marked done but wasn't added!
    """


class AlreadyDoneError(SchedulerError, ValueError):
    """
    Node was marked done, but it was already done!
    """
