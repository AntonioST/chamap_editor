from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from neurocarto.typing import *
from .surrounding import surrounding, get_surrounding

if TYPE_CHECKING:
    from neurocarto.util.util_blueprint import BlueprintFunctions

__all__ = ['ClusteringEdges', 'find_clustering', 'clustering_edges', 'edge_rastering']


class ClusteringEdges(NamedTuple):
    """
    Edge of area.
    """

    category: int
    """electrode category value"""
    shank: int
    """shank number"""
    edges: list[tuple[int, int, int]]  # [(x, y, corner)]
    """
    corner code::

        3 2 1
        4 8 0
        5 6 7

    origin at bottom left.

    """

    @property
    def x(self) -> Array[int, N]:
        return np.array([it[0] for it in self.edges])

    @property
    def y(self) -> Array[int, N]:
        return np.array([it[1] for it in self.edges])

    def with_shank(self, s: int) -> ClusteringEdges:
        return self._replace(shank=s)

    def with_category(self, c: int) -> ClusteringEdges:
        return self._replace(category=c)

    def set_corner(self, tr: tuple[int, int],
                   tl: tuple[int, int] = None,
                   bl: tuple[int, int] = None,
                   br: tuple[int, int] = None) -> ClusteringEdges:
        if tl is None and bl is None and br is None:
            w, h = tr
            tl = -w, h
            bl = -w, -h
            br = w, -h

        offset = [None, tr, None, tl, None, bl, None, br, (0, 0)]
        edges = [
            (x + off[0], y + off[1], 8)
            for x, y, c in self.edges
            # corner at 0, 2, 4, 6 are removed
            if (off := offset[c]) is not None
        ]
        return self._replace(edges=edges)


def find_clustering(self: BlueprintFunctions,
                    blueprint: Array[int, A],
                    categories: int | list[int] = None, *,
                    diagonal=True) -> Array[int, A]:
    """
    find electrode clustering with the same category.

    :param self:
    :param blueprint: Array[category, N]
    :param categories: only for given categories.
    :param diagonal: does surrounding includes electrodes on diagonal?
    :return: Array[int, N]
    """
    if len(blueprint) != len(self.s):
        raise ValueError()

    if isinstance(categories, int):
        categories = [categories]

    ret: Array[int, A] = np.arange(len(blueprint)) + 1

    unset = self.CATE_UNSET
    ret[blueprint == unset] = 0
    if categories is not None:
        for category in np.unique(blueprint):
            if int(category) not in categories:
                ret[blueprint == category] = 0

    def union(i: int, j: int):
        if i == j:
            return

        a: int = ret[i]
        b: int = ret[j]
        c = min(a, b)

        if a != c:
            ret[ret == a] = c
        if b != c:
            ret[ret == b] = c

    for i in range(len(blueprint)):
        if ret[i] > 0:
            for j in surrounding(self, i, diagonal=diagonal):
                if blueprint[i] == blueprint[j]:
                    union(i, j)

    return ret


def clustering_edges(self: BlueprintFunctions,
                     blueprint: Array[int, A],
                     categories: int | list[int] = None) -> list[ClusteringEdges]:
    """
    For each clustering block, calculate its edges.

    :param blueprint:
    :param categories:
    :return: list of ClusteringEdges
    """
    dx = self.dx
    dy = self.dy

    clustering = find_clustering(self, blueprint, categories, diagonal=False)
    ret = []

    for cluster in np.unique(clustering):
        if cluster == 0:
            continue

        area: Array[bool, A] = clustering == cluster

        c = np.unique(blueprint[area])
        assert len(c) == 1
        c = int(c[0])

        s = np.unique(self.s[area])
        assert len(s) == 1
        s = int(s[0])

        x = self.x[area]
        y = self.y[area]

        if np.count_nonzero(area) == 1:
            ret.append(ClusteringEdges(c, s, [(x, y, 1), (x, y, 3), (x, y, 5), (x, y, 7)]))
        else:
            x0 = int(np.min(x))
            y0 = int(np.min(y[x == x0]))

            i = self._position_index[(s, int(x0 / dx), int(y0 / dy))]
            ret.append(ClusteringEdges(c, s, _cluster_edge(self, area, i)))

    return ret


def _cluster_edge(self: BlueprintFunctions,
                  area: Array[bool, N],
                  i: int) -> list[tuple[int, int, int]]:
    """

    :param area:
    :param i: start index
    :return: list of (x, y, corner)
    """
    if not area[i]:
        raise ValueError(f'no cluster at index {i}')

    pos = self._position_index

    # 3 2 1
    # 4 e 0
    # 5 6 7

    actions = {
        # direction (i -> j):
        0: {  # rightward
            # next direction (j -> k): corner
            # None: (corners), action

            # * * *
            # i j *
            #   k *
            6: 5,

            # * * *
            # i j k
            # ?
            0: 6,

            # * k
            # i j
            #
            2: 7,

            # ?
            # i j
            #
            None: ((7, 1, 3), 2)
        },
        6: {  # downward
            #   i *
            # k j *
            # * * *
            4: 3,

            # ? i *
            #   j *
            #   k *
            6: 4,

            # ? i *
            #   j k
            #
            0: 5,

            # ? i ?
            #   j
            #
            None: ((5, 7, 1), 0)
        },
        4: {  # leftward
            # ? k
            # * j i
            # * * *
            2: 1,

            #
            # k j i
            # * * *
            4: 2,

            #
            #   j i
            # ? k *
            6: 3,

            #
            #   j i
            #     ?
            None: ((3, 5, 7), 6)
        },
        2: {  # upward
            # * * *
            # * j k
            # * i
            0: 7,

            # * k
            # * j
            # * i
            2: 0,

            #
            # k j
            # * i
            4: 1,

            #
            #   j
            # ? i
            None: ((1, 3, 5), 4)
        }
    }

    x = self.x[i]
    y = self.y[i]
    ret = [(x, y, 5)]
    # * ?
    # i * ?
    #   ? ?
    j = i
    d = 0  # right
    while not (i == j and d == 6):
        if not area[j]:
            raise ValueError(f'no cluster at index j={j}')

        # print(debug_print_local(self, area.astype(int), j, size=2))
        x = self.x[j]
        y = self.y[j]
        for action, corner in actions[d].items():
            if action is not None:
                if (k := pos.get(get_surrounding(self, j, action), None)) is not None and area[k]:
                    ret.append((x, y, corner))
                    j = k
                    d = action
                    break
                else:
                    continue
            elif action is None:
                corner, action = corner
                for _corner in corner:
                    ret.append((x, y, _corner))
                d = action
                break
        else:
            raise RuntimeError('un-reachable')

    return ret


def edge_rastering(self: BlueprintFunctions,
                   edges: ClusteringEdges | list[ClusteringEdges], *,
                   fill=False,
                   overwrite=False) -> Array[int, A]:
    """
    For given edges, put them on the blueprint.

    :param edges:
    :param fill: fill the area.
    :param overwrite: latter result overwrite previous results
    :return: blueprint
    """
    match edges:
        case (ClusteringEdges() as edge) | [ClusteringEdges() as edge]:
            return _edge_rastering(self, edge, fill=fill)

    unset = self.CATE_UNSET
    ret = np.full_like(self.s, unset)
    for edge in edges:
        res = _edge_rastering(self, edge, fill=fill)
        if overwrite:
            ret = np.where(res == unset, ret, res)
        else:
            ret = np.where(ret == unset, res, ret)
    return ret


def _edge_rastering(self: BlueprintFunctions, edge: ClusteringEdges, fill=False) -> Array[int, A]:
    dx = self.dx
    dy = self.dy
    pos = self._position_index

    c = edge.category
    s = edge.shank
    edge = edge.set_corner((0, 0))
    edge = [*edge.edges, edge.edges[0]]  # as closed polygon

    unset = self.CATE_UNSET
    ret = np.full_like(self.s, unset)

    for i in range(len(edge) - 1):
        px, py, _ = edge[i]
        qx, qy, _ = edge[i + 1]
        if px == qx:
            x = int(px / dx)
            py, qy = min(py, qy), max(py, qy)
            for y in range(py, qy + 1):
                y = int(y / dy)
                if (p := pos.get((s, x, y), None)) is not None:
                    ret[p] = c
        elif py == qy and not fill:
            y = int(py / dy)
            px, qx = min(px, qx), max(px, qx)
            for x in range(px, qx + 1):
                x = int(x / dx)
                if (p := pos.get((s, x, y), None)) is not None:
                    ret[p] = c

    if not fill:
        return ret

    n_x = len(set([
        x for _s, x, y in self._position_index
        if _s == s
    ]))

    ret = ret.reshape((-1, n_x))  # (Y, X)
    interior = np.cumsum((ret != unset).astype(int), axis=1) % 2 == 1
    ret[interior] = c

    return ret.ravel()
