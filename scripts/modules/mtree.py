from math import inf

class MTree:
    class Node:    
        def __init__(self, point):
            self._point = point
            self._left = self._right = None
            self._radius = 0
            
        def rebalance(self, dist_func):
            lc_dist = dist_func(self._point, self._left._point)
            rc_dist = dist_func(self._point, self._right._point)
            lr_dist = dist_func(self._left._point, self._right._point)
            if lc_dist > rc_dist and lc_dist > lr_dist:
                self._point, self._right._point = self._right._point, self._point
                self._radius = max(rc_dist, lr_dist)
            elif rc_dist > lc_dist and rc_dist > lr_dist:
                self._point, self._left._point = self._left._point, self._point
                self._radius = max(lc_dist, lr_dist)
        
        def add_point(self, point, dist_func, distance = None):
            if distance == None:
                distance = dist_func(point, self._point)
            self._radius = max(self._radius, distance)
            if self._left == None:
                self._left = MTree.Node(point)
            elif self._right == None:
                self._right = MTree.Node(point)
                self.rebalance(dist_func)
            else:
                left_dist = dist_func(point, self._left._point)
                right_dist = dist_func(point, self._right._point)
                if left_dist < right_dist:
                    self._left.add_point(point, dist_func, left_dist)
                else:
                    self._right.add_point(point, dist_func, right_dist)
         
        def get_nearest(self, point, dist_func, best_dist, best_value, distance = None):
            if distance == None:
                distance = dist_func(point, self._point)
            if distance < best_dist:
                best_dist, best_value = distance, self._point
                
            left_dist = dist_func(point, self._left._point) if self._left != None else inf
            right_dist = dist_func(point, self._right._point) if self._right != None else inf
            if left_dist < right_dist:
                if self._left != None and self._left._radius + best_dist > left_dist:
                    left_dist, left_value = self._left.get_nearest(point, dist_func, best_dist, best_value, left_dist)
                    if left_dist < best_dist:
                        best_dist, best_value = left_dist, left_value
                if self._right != None and self._right._radius + best_dist > right_dist:
                    right_dist, right_value = self._right.get_nearest(point, dist_func, best_dist, best_value, right_dist)
                    if right_dist < best_dist:
                        best_dist, best_value = right_dist, right_value
            else:
                if self._right != None and self._right._radius + best_dist > right_dist:
                    right_dist, right_value = self._right.get_nearest(point, dist_func, best_dist, best_value, right_dist)
                    if right_dist < best_dist:
                        best_dist, best_value = right_dist, right_value
                if self._left != None and self._left._radius + best_dist > left_dist:
                    left_dist, left_value = self._left.get_nearest(point, dist_func, best_dist, best_value, left_dist)
                    if left_dist < best_dist:
                        best_dist, best_value = left_dist, left_value
            return best_dist, best_value
        
        def __repr__(self, w=""):
            return f"{self._point} ({self._radius})\n{w}+>{self._right.__repr__(w + '| ') if self._right else 'None'}\n{w}+>{self._left.__repr__(w + '  ') if self._left else 'None'}"

    def __init__(self, dist_func):
        self._root = None
        self._dist_func = dist_func
        
    def add_point(self, point):
        if self._root == None:
            self._root = MTree.Node(point)
        else:
            self._root.add_point(point, self._dist_func)

    def get_nearest(self, point):
        if self._root == None:
            return inf, None
        else:
            return self._root.get_nearest(point, self._dist_func, inf, None)