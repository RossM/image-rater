from math import inf
from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor

class MTree:
    class Node:
        _points: Tensor
        _children: List['MTree.Node']
        _radius: Tensor
        _count: int

        def __init__(self, points: Tensor):
            self._points = points
            self._children = []
            self._radius = torch.tensor(0, dtype=points.dtype)
            self._count = points.shape[0]
        
        def add_point(self, tree: 'MTree', point: Tensor):
            self._points = torch.cat((self._points, point[None, :]), dim=0)
            self._radius.clamp_(min=tree._dist_func(self._points[0], point))
            self._count += 1

        def get_nearest(self, tree: 'MTree', point: Tensor, best_dist: Tensor, best_point: Tensor) -> Tuple[Tensor, Tensor]:
            distances = tree._dist_func(self._points, point[None, :])
            my_dist, my_idx = distances.min(dim=0)
            if my_dist < best_dist:
                best_dist, best_point = my_dist, self._points[my_idx]
          
            for child in self._children:
                child_dist = tree._dist_func(child._points[0], point)
                if child_dist < child._radius + best_dist:
                    best_dist, best_point = child.get_nearest(tree, point, best_dist, best_point)
                    
            return best_dist, best_point


    _root: 'MTree.Node'
    _dist_func: Callable[[Tensor, Tensor], Tensor]

    def __init__(self, dist_func: Callable[[Tensor, Tensor], Tensor] = None):
        def torch_distance(x: Tensor, y: Tensor) -> Tensor:
            return (x - y).norm(dim=-1)

        self._root = None
        self._dist_func = dist_func or torch_distance
        
    @torch.no_grad()
    def add_point(self, point: Tensor | list[float]):
        if isinstance(point, list):
            point = torch.tensor(point)

        if self._root == None:
            self._root = MTree.Node(point[None,:])
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
