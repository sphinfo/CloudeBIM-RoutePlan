# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.


class InputDataError(Exception):
    """Exception raised for input data error."""

    def __init__(self, message: str):
        self.message = message


class RouteCreationError(Exception):
    """Exception raised for route plan creation error."""

    def __init__(self, message: str):
        self.message = message