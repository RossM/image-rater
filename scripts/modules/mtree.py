from math import inf
from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor


class MTree:
    class Node:
        _points: Tensor
        _children: List["MTree.Node"]
        _radii: Tensor
        _count: int

        def __init__(self, points: Tensor) -> None:
            self._points = points
            self._children = None
            self._radii = None
            self._count = points.shape[0]

        def add_point(self, tree: "MTree", point: Tensor) -> None:
            self._count += 1
            if self._points.shape[0] < tree._max_node_size:
                self._points = torch.cat((self._points, point[None, :]), dim=0)
            else:
                if self._children == None:
                    self._children = [None] * self._points.shape[0]
                    self._radii = self._points.new_zeros((self._points.shape[0]))

                _, index = tree._dist_func(point[None, :], self._points).min(dim=0)
                if self._children[index] == None:
                    self._children[index] = MTree.Node(point[None, :])
                else:
                    self._children[index].add_point(tree, point)
                    self._radii[index].clamp_(min=tree._dist_func(self._points[0], point))

        def get_nearest(
            self, tree: "MTree", point: Tensor, best_dist: Tensor, best_point: Tensor
        ) -> Tuple[Tensor, Tensor]:
            distances = tree._dist_func(self._points, point[None, :])
            _, indexes = distances.sort()

            if distances[indexes[0]] < best_dist:
                best_dist, best_point = distances[indexes[0]], self._points[indexes[0]]

            if self._children != None:
                for index in indexes:
                    child = self._children[index]
                    if child != None and distances[index] < self._radii[index] + best_dist:
                        best_dist, best_point = child.get_nearest(
                            tree, point, best_dist, best_point
                        )

            return best_dist, best_point

    _root: "MTree.Node"
    _dist_func: Callable[[Tensor, Tensor], Tensor]
    _max_node_size: int

    def __init__(
        self,
        dist_func: Callable[[Tensor, Tensor], Tensor] = None,
        max_node_size: int = 64,
    ):
        def torch_distance(x: Tensor, y: Tensor) -> Tensor:
            return (x - y).norm(dim=-1)

        self._root = None
        self._dist_func = dist_func or torch_distance
        self._max_node_size = max_node_size

    @torch.no_grad()
    def add_point(self, point: Tensor | list[float]):
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            self._root = MTree.Node(point[None, :])
        else:
            self._root.add_point(self, point)

    @torch.no_grad()
    def get_nearest(self, point: Tensor | list[float]) -> Tuple[float, Tensor]:
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            return inf, None
        else:
            dist, point = self._root.get_nearest(self, point, torch.tensor(inf), None)
            return dist.item(), point
