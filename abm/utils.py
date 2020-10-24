class Sequence(object):
    """
    Monotonically increasing counter for assigning autoincrementing ids.
    """

    def __init__(self):
        self._counter = 0

    def __call__(self):
        self._counter += 1
        return self._counter

    def __iter__(self):
        while True:
            yield self()
