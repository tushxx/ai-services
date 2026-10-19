class MeterMindError(Exception):
    pass


class UnsupportedCategoryError(MeterMindError):
    pass


class ReadingStorageError(MeterMindError):
    pass


class InvalidReadingError(MeterMindError):
    pass


class ToolError(MeterMindError):
    """Raised where a tool cannot answer."""
    pass


class NoSuchThread(MeterMindError):
    """Raised where a thread is asked about that nobody started, or that is not waiting for anybody."""
    pass
