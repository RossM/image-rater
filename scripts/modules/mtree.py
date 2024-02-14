from math import inf
from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor


def k_means(points: Tensor, groups: int, rounds: int = 3) -> Tensor:
    # Get initial centerpoints
    centers = points.new_zeros(groups, points.shape[1])
    for i, chunk in enumerate(points.chunk(groups)):
        centers[i] = chunk.mean(dim=0)

    # Run k-means
    for _ in range(rounds):
        distances = torch.cdist(centers, points)
        group_index = distances.argmin(dim=0)
        for i in range(groups):
            centers[i] = points[group_index == i].mean(dim=0)

    distances = torch.cdist(centers, points)
    group_index = distances.argmin(dim=0)

    # Find the point that minimizes the radius of each group
    for i in range(groups):
        group_points = points[group_index == i]
        radius, _ = torch.cdist(group_points, group_points).max(dim=1)
        centers[i] = group_points[radius.argmin(dim=0)]

    return centers


class MTree:
    class Node:
        _points: Tensor
        _children: List["MTree.Node"]
        _radii: Tensor
        _count: int

        def __init__(self, points: Tensor, capacity: int) -> None:
            self._points = points.new_zeros((capacity, points.shape[1]))
            self._points[: points.shape[0]] = points
            self._children = None
            self._radii = None
            self._count = points.shape[0]

        def split(self, tree: "MTree") -> None:
            points = self._points
            self._points = k_means(points, groups=tree._branching)
            self._children = [None] * self._points.shape[0]
            self._radii = points.new_zeros((self._points.shape[0]))

            distances = torch.cdist(self._points, points)
            group_idx = distances.argmin(dim=0)
            for i in range(tree._branching):
                self._children[i] = MTree.Node(
                    points[group_idx == i], tree._max_node_size
                )
                self._radii[i] = distances[group_idx, i].max()

        def add_point(self, tree: "MTree", point: Tensor) -> None:
            self._count += 1
            if self._children == None:
                self._points[self._count - 1] = point
                if self._count >= tree._max_node_size:
                    self.split(tree)
            else:
                dist, index = tree._dist_func(point[None, :], self._points).min(dim=0)
                self._children[index].add_point(tree, point)
                self._radii[index].clamp_(min=dist)

        def get_nearest(
            self, tree: "MTree", point: Tensor, best_dist: Tensor, best_point: Tensor
        ) -> Tuple[Tensor, Tensor]:
            distances = tree._dist_func(self._points[: self._count], point[None, :])

            if self._children == None:
                new_dist, index = distances.min(dim=0)
                if new_dist < best_dist:
                    best_dist, best_point = new_dist, self._points[index]
            else:
                _, indexes = distances.sort()
                for index in indexes:
                    if distances[index] < self._radii[index] + best_dist:
                        best_dist, best_point = self._children[index].get_nearest(
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
        max_node_size: int = 16384,
        branching: int = 4,
    ):
        def torch_distance(x: Tensor, y: Tensor) -> Tensor:
            return (x - y).norm(dim=-1)

        self._root = None
        self._dist_func = dist_func or torch_distance
        self._max_node_size = max_node_size
        self._branching = min(branching, max_node_size)

    @torch.no_grad()
    def add_point(self, point: Tensor | list[float]):
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            self._root = MTree.Node(point[None, :], self._max_node_size)
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
