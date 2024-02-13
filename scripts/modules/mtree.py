from math import inf

class MTree:
    class Node:    
        def __init__(self, point):
            self._point = point
            self.reset()
           
        def reset(self):
            self._left = self._right = None
            self._radius = 0
            self._count = 1
            
        def rebalance(self, dist_func, rebalance_depth):
            for r in range(3):
                dist_lc = dist_func(self._point, self._left._point)
                dist_rc = dist_func(self._point, self._right._point)
                dist_lr = dist_func(self._left._point, self._right._point)
                if dist_rc > dist_lc and dist_rc > dist_lr:
                    new_parent = self._left
                    children = [self._left._left, self._left._right, self._right, MTree.Node(self._point)]
                elif dist_lc > dist_rc and dist_lc > dist_lr:
                    new_parent = self._right
                    children = [self._right._left, self._right._right, self._left, MTree.Node(self._point)]
                else:
                    return
            
                children = list((dist_func(new_parent._point, child._point), child) for child in children if child != None)
                children.sort(key=lambda c: c[0], reverse=True)
            
                if children[0][0] > self._radius:
                    # The new root would end up with a larger radius, abort rebalancing
                    return
            
                #print(f"Before rebalance (round {r} depth {rebalance_depth}):\n{self}")
                self.reset()
                self._point = new_parent._point
                for distance, child in children:
                    self.add_node(child, dist_func, distance, rebalance_depth+1)
                #print(f"After rebalance (round {r} depth {rebalance_depth}):\n{self}")
        
        def add_node(self, node, dist_func, distance = None, rebalance_depth=0):
            if node == None:
                return
            
            self._count += node._count

            if node._count < 100:
                self._radius = max(self._radius, max(dist_func(point, self._point) for point in iter(node)))
            else:
                if distance == None:
                    distance = dist_func(node._point, self._point)              
                self._radius = max(self._radius, distance + node._radius)
            
            if self._left == None:
                self._left = node
            elif self._right == None:
                self._right = node
            elif self._left._count + 5 <= self._right._count:
                self._left.add_node(node, dist_func, None, rebalance_depth)
            elif self._right._count + 5 <= self._left._count:
                self._right.add_node(node, dist_func, None, rebalance_depth)
            else:
                left_dist = dist_func(node._point, self._left._point)
                right_dist = dist_func(node._point, self._right._point)
                if left_dist < right_dist:
                    self._left.add_node(node, dist_func, left_dist, rebalance_depth)
                else:
                    self._right.add_node(node, dist_func, right_dist, rebalance_depth)

            if rebalance_depth < 5 and self._left != None and self._right != None:
                self.rebalance(dist_func, rebalance_depth)
               
            #assert(self._radius >= max(dist_func(point, self._point) for point in iter(self)))
         
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
            return f"{self._point} ({self._radius})\n{w}+-{self._right.__repr__(w + '| ') if self._right else 'None'}\n{w}+-{self._left.__repr__(w + '  ') if self._left else 'None'}"
        
        def __iter__(self):
            yield self._point
            if self._left != None:
                yield from self._left
            if self._right != None:
                yield from self._right

    def __init__(self, dist_func):
        self._root = None
        self._dist_func = dist_func
        
    def add_point(self, point):
        if self._root == None:
            self._root = MTree.Node(point)
        else:
            self._root.add_node(MTree.Node(point), self._dist_func)

    def get_nearest(self, point):
        if self._root == None:
            return inf, None
        else:
            return self._root.get_nearest(point, self._dist_func, inf, None)