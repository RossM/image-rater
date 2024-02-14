from math import inf
from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor


def k_means(points: Tensor, groups: int, rounds: int = 3) -> Tensor:
    # Get initial centerpoints
    centers = points.new_zeros(groups, points.shape[1])
    for i, chunk in enumerate(points.chunk(groups)):
        centers[i] = chunk.mean(dim=0)

    for _ in range(rounds):
        distances = torch.cdist(centers, points)
        group_index = distances.argmin(dim=0)
        for i in range(groups):
            centers[i] = points[group_index == i].mean(dim=0)

    distances = torch.cdist(centers, points)
    group_index = distances.argmin(dim=0)

    reps = points.new_zeros(groups, points.shape[1])

    for i in range(groups):
        group_points = points[group_index == i]
        worst_dist, _ = torch.cdist(group_points, group_points).max(dim=1)
        reps[i] = group_points[worst_dist.argmin(dim=0)]

    return reps


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
            if self._children == None:
                if self._points.shape[0] < tree._max_node_size:
                    self._points = torch.cat((self._points, point[None, :]), dim=0)
                else:
                    points = self._points
                    self._points = k_means(points, groups=tree._branching)
                    self._children = [None] * self._points.shape[0]
                    self._radii = points.new_zeros((self._points.shape[0]))

                    for point in points:
                        self.add_point(tree, point)
            else:
                dist, index = tree._dist_func(point[None, :], self._points).min(dim=0)
                if self._children[index] == None:
                    self._children[index] = MTree.Node(point[None, :])
                else:
                    self._children[index].add_point(tree, point)
                    self._radii[index].clamp_(min=dist)

        def get_nearest(
            self, tree: "MTree", point: Tensor, best_dist: Tensor, best_point: Tensor
        ) -> Tuple[Tensor, Tensor]:
            distances = tree._dist_func(self._points, point[None, :])

            if self._children == None:
                _, index = distances.min(dim=0)
                if distances[index] < best_dist:
                    best_dist, best_point = distances[index], self._points[index]
            else:
                _, indexes = distances.sort()
                for index in indexes:
                    child = self._children[index]
                    if (
                        child != None
                        and distances[index] < self._radii[index] + best_dist
                    ):
                        best_dist, best_point = child.get_nearest(
                            tree, point, best_dist, best_point
                        )

            return best_dist, best_point

    _root: "MTree.Node"
    _dist_func: Callable[[Tensor, Tensor], Tensor]
    _max_node_size: int
    _branching: int

    def __init__(
        self,
        dist_func: Callable[[Tensor, Tensor], Tensor] = None,
        max_node_size: int = 4096,
        branching: int = 8,
    ):
        def torch_distance(x: Tensor, y: Tensor) -> Tensor:
            return (x - y).norm(dim=-1)

        self._root = None
        self._dist_func = dist_func or torch_distance
        self._max_node_size = max_node_size
        self._branching = branching

    @torch.no_grad()
    def add_point(self, point: Tensor | list[float]):
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            self._root = MTree.Node(point[None, :])
        else:
            self._root.add_point(self, point)

    @torch.no_grad()
    def get_nearest(
        self, point: Tensor | list[float], max_dist: float = inf
    ) -> Tuple[float, Tensor]:
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            return max_dist, None
        else:
            dist, point = self._root.get_nearest(
                self, point, torch.tensor(max_dist), None
            )
            return dist.item(), point
