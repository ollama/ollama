class SetupError(Exception):
    pass


class WaitTimeoutError(Exception):
    pass


class RequestError(Exception):
    pass


class UnknownValueError(Exception):
    pass


class TranscriptionNotReady(Exception):
    pass


class TranscriptionFailed(Exception):
    pass
