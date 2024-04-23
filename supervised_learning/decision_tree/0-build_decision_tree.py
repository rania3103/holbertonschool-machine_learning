#!/usr/bin/env python3
"""Depth of a decision tree"""
import numpy as np


class Node:
    """Node class"""

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """returns the maximum of the depths of the nodes
        (including the leaves) in a decision tree"""
        max_depth_right = 0
        right_subtree = self
        while (right_subtree.right_child):
            if right_subtree.right_child:
                right_subtree = right_subtree.right_child
            else:
                right_subtree = right_subtree.left_child
            max_depth_right += 1
        max_depth_left = 0
        left_subtree = self
        while (left_subtree.left_child):
            if left_subtree.left_child:
                left_subtree = left_subtree.left_child
            else:
                left_subtree = left_subtree.right_child
            max_depth_left += 1
        return max(max_depth_left, max_depth_right)


class Leaf(Node):
    """Leaf class that inherits from Node class"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth


class Decision_Tree():
    """Decision_Tree class"""

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()
