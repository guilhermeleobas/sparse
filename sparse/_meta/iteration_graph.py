from typing import Tuple
from collections import defaultdict


class IndexVar(object):
    def __init__(self, name):
        assert name is not None
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return f"index({self._name})"

    def __repr__(self):
        return self.__str__()


class Node(IndexVar):
    def __init__(self, *, name=None):
        assert name is not None
        super().__init__(name)

    @property
    def root(self):
        return False

    @classmethod
    def from_iv(cls, iv):
        assert isinstance(iv, IndexVar)
        return cls(name=iv.name)


class RootNode(Node):
    def __init__(self, *, name):
        super().__init__(name=name)

    @property
    def root(self):
        return True

    def __str__(self):
        return f"root({self.name})"


class Edge(object):
    def __init__(self, u, v, label=None):
        assert isinstance(u, Node)
        assert isinstance(v, Node)
        self.u = u
        self.v = v
        self.label = label


class Access(object):
    def __init__(self, *args, name=None):
        self._elems = tuple(args)
        self._name = name

    @classmethod
    def from_numpy_notation(cls, *, notation: Tuple, name: str):

        char_code = 105  # 'i'
        _vars = []

        for e in notation:
            if isinstance(e, slice):
                # slice(...)
                iv = IndexVar(chr(char_code))
                char_code += 1
                _vars.append(iv)
            elif e is None:
                char_code += 1
            else:
                raise NotImplementedError(e)
        return Access(*_vars, name=name)

    def __add__(self, other):
        return IterationGraph(self, other)

    def __sub__(self, other):
        return IterationGraph(self, other)

    def __mul__(self, other):
        return IterationGraph(self, other)

    def __div__(self, other):
        return IterationGraph(self, other)

    def __iter__(self):
        yield from self._elems

    def __str__(self):
        s = f"Access[{self.name}]"
        ivs = tuple(map(lambda x: x.name, self._elems))
        s += "(" + ", ".join(ivs) + ")"
        return s

    @property
    def name(self):
        return self._name

    def iterwise(self):
        root = RootNode(name=self.name)
        node = Node.from_iv(self._elems[0])
        yield Edge(root, node, label=f"{self.name}1")

        for idx in range(1, len(self._elems)):
            node2 = Node.from_iv(self._elems[idx])
            yield Edge(node, node2, label=f"{self.name}{idx+1}")
            node = node2


class IterationGraph(object):
    def __init__(self, *args):
        self._graph = defaultdict(list)
        for access in args:
            assert isinstance(access, Access)
            self._iterate_in_access(access)

    def _iterate_in_access(self, access):
        for edge in access.iterwise():
            assert isinstance(edge, Edge)
            u = edge.u
            self._graph[u].append(edge)

    def has_cycle(self):
        raise NotImplementedError()

    def root(self):
        raise NotImplementedError()

    def __str__(self):
        s = ""
        for u, l in self._graph.items():
            for edge in l:
                v = edge.v
                label = edge.label
                s += f"{u} ~~{label}~~> {v}\n"
        return s

    def _to_graphviz(self):
        from graphviz import Digraph

        g = Digraph("G")
        for u, l in self._graph.items():
            for edge in l:
                u = edge.u.name
                v = edge.v.name
                label = edge.label
                g.edge(u, v, label)
        return g

    def view(self):
        self._to_graphviz().view()

    def to_dot(self):
        return self._to_graphviz().source
