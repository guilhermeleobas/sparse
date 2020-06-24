from typing import Tuple
from .iteration_graph import Access


class Format(object):
    def __init__(self, *, name: str, levels: Tuple):
        self._levels = levels
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def levels(self):
        return self._levels

    def __str__(self):
        levels_str = tuple(map(lambda x: str(x.__name__), self.levels))
        s = f"{self.name}"
        s += "(" + ", ".join(levels_str) + ")"
        return s

    def __len__(self):
        return len(self.levels)


class LazyTensor(object):
    def __init__(self, *, dims: Tuple, format: Format):
        assert len(dims) == len(format)
        self._dims = dims
        self._format = format

    @property
    def dims(self):
        return self._dims

    @property
    def format(self):
        return self._format

    def __str__(self):
        format_name = self.format.name
        s = f"{self.format.name}"
        z = zip(self.dims, self.format.levels)
        s += "(" + ", ".join(f"{level.__name__}[{dim}]" for dim, level in z) + ")"
        return s

    def __getitem__(self, t):
        if isinstance(t, slice):
            t = (t,)

        return Access.from_numpy_notation(notation=t, name=self.format.name)


class Tensor(LazyTensor):
    pass
