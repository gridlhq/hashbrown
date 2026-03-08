import os

VERSION = 1

def helper(value: int) -> int:
    return value + 1

@logged
@timer()
def decorated(value: int) -> int:
    return value * 2

class Box:
    def method(self, value: int):
        return value
