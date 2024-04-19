import logging
from typing import Any, Callable, Optional, Sequence


logger = logging.getLogger(__name__)


class Node:
    """Node class"""

    parent: Optional["Node"]
    children: Sequence["Node"]
    data: dict[str, Any]

    def __init__(self, parent: Optional["Node"] = None):
        self.parent = parent
        self.children = []
        self.data = {}

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.children[i]
        i, *rest = i
        return self.children[i][rest] if rest else self.children[i]

    def __iter__(self):
        return iter(self.children)

    def depth(self):
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def add_data(self, **data):
        self.data |= data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def leaves(self):
        """Return the leaf nodes attached to this node"""
        return self.traverse(filter=lambda node: node.is_leaf())

    def traverse(self, filter: Optional[Callable[["Node"], bool]] = None) -> Sequence["Node"]:
        """Return all nodes attached to this node"""
        nodes = [self] if (filter is None or filter(self)) else []
        for child in self.children:
            nodes.extend(child.traverse(filter=filter))
        return nodes

    def tree2str(self, sep: str = "", is_last: bool = True) -> str:
        """Return a string representation of this node and its decendants"""
        lines = [sep + ("└──" if is_last else "├──") + str(self)]
        sep += "    " if is_last else "│   "
        for child in self.children:
            lines.append(child.tree2str(sep, child == self.children[-1]))
        return "\n".join(lines).strip("└──")

    def is_leaf(self) -> bool:
        """True if this node does not have any children"""
        return not self.children

    def asdict(self):
        """This node's and its decendents' data as a dictionary"""
        if self.is_leaf():
            return self.data
        return self.data | {"contains": [child.asdict() for child in self.children]}

    def detach(self):
        """Detach node from tree

        Removes the node from its parent's children and sets its parent
        to None, effectively removing it from the tree.
        """
        if self.parent:
            siblings = self.parent.children
            self.parent.children = [child for child in siblings if child != self]
        self.parent = None

    def prune(self, condition: Callable[["Node"], bool], include_starting_node=True):
        """Prune the tree

        Removes (detaches) all nodes starting from this node that
        fulfil the given condition. Any decendents of a node that
        fulfils the condition are also removed.

        Arguments:
            condition: A function `f` where `f(node) == True` if `node`
                should be removed from the tree.
            include_starting_node: Whether to include the starting node
                or not. If False, the starting node will not be
                detached from its parent even though it fulfils the
                given condition. Defaults to True.

        Example: To remove all nodes at depth 2, use
            node.prune(lambda node: node.depth() == 2)
        """
        nodes = self.traverse(filter=condition)
        for node in nodes:
            if not include_starting_node and node == self:
                continue
            node.detach()
        logger.info("Removed %d nodes from the tree", len(nodes))

    def max_depth(self):
        """Return the max depth of the tree starting at this node"""
        return max(node.depth() for node in self.leaves())
