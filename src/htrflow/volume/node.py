import logging
from collections import defaultdict
from itertools import count
from typing import Any, Callable, Iterable, Iterator, Sequence

from typing_extensions import Self


logger = logging.getLogger(__name__)


class Node:
    """Node class

    This class represents a single element of a tree. It has methods
    for traversing and modifying the tree starting at this node. Its
    attributes are:

    Attributes:
        parent: The parent node of this node. Is None if this node is
            a root node.
        children: A list of child nodes attached to this node. The list
            may be empty; the node is then said to be a leaf node.
        data: A dictionary containing all data associated with this
            node. It can be seen as the actual content of the node, while
            the two former attributes define the node's position in the
            tree.
    """

    parent: Self | None
    children: list[Self]
    data: dict[str, Any]

    def __init__(self, parent: Self | None = None, label: str | None = None):
        self.parent = parent
        self.children = []
        self.data = {}
        self.depth = 0 if parent is None else parent.depth + 1

        self._label = label
        self._long_label = None

    @property
    def label(self):
        return self._long_label or self._label

    def relabel(self):
        """Relabel the tree

        Assigns unique labels to this node and its children based on
        the labels given at initialization and their position in the tree.

        root              root
        ├── node    =>    ├── root_node0
        └── node          └── root_node1

        """
        counter = defaultdict(lambda: count(0))
        for child in self:
            idx = next(counter[child._label])
            child._long_label = f"{self.label}_{child._label}{idx}"
            child.relabel()

    def __getitem__(self, i: int | Iterable[int]):
        """Enables tuple indexing of tree

        Examples:
            node[0] returns the first child of this node
            node[0, 1] is equal to node[0][1] and returns the second
                child of the first child of this node
        """
        if isinstance(i, int):
            return self.children[i]
        i, *rest = i
        return self.children[i][rest] if rest else self.children[i]

    def __iter__(self) -> Iterator["Node"]:
        """Enables the syntax `for child in node`"""
        return iter(self.children)

    def add_data(self, **data) -> None:
        """Add data to node

        Example: node.add_data(label="my_node") will save the key-value
        pair ("label": "my_node") to the node. To retrieve the value,
        use `node.get("label")`.

        Arguments:
            **data: Key-value pairs of data to add to `node`. Overwrites
                the old value if the key is already present.
        """
        self.data |= data

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from node

        Returns the value set by a previous call to
        node.add_data(key=...) if present, else `default`.

        Arguments:
            key: Key of value to retrieve
            default: A default value to be returned if `key` is not
                present in the node.
        """
        return self.data.get(key, default)

    def leaves(self) -> Sequence[Self]:
        """Return the leaf nodes of the tree"""
        return self.traverse(filter=lambda node: node.is_leaf())

    def traverse(self, filter: "Callable[[Node], bool] | None" = None) -> Sequence[Self]:
        """Return all nodes attached to this node, including self

        Arguments:
            filter: An optional filtering function. If passed, only
                nodes where `filter(node) == True` will be returned.
        """
        nodes = [self] if (filter is None or filter(self)) else []
        for child in self.children:
            nodes.extend(child.traverse(filter=filter))
        return nodes

    def tree2str(self, sep: str = "", is_last: bool = True) -> str:
        """Return a string representation of this node and its decendents"""
        lines = [sep + ("└──" if is_last else "├──") + str(self)]
        sep += "    " if is_last else "│   "
        for child in self.children:
            lines.append(child.tree2str(sep, child == self.children[-1]))
        return "\n".join(lines).strip("└──")

    def is_leaf(self) -> bool:
        """True if this node does not have any children"""
        return not self.children

    def asdict(self) -> dict[str, Any]:
        """Return the tree as a nested dicionary"""
        data = self.data | {"label": self.label}
        if self.is_leaf():
            return data
        return data | {"contains": [child.asdict() for child in self.children]}

    def detach(self) -> None:
        """Detach node from tree

        Removes the node from its parent's children and sets its parent
        to None, effectively removing it from the tree.
        """
        if self.parent:
            siblings = self.parent.children
            self.parent.children = [child for child in siblings if child != self]
        self.parent = None

    def prune(self, condition: Callable[["Node"], bool], include_starting_node: bool = True) -> None:
        """Prune the tree

        Removes (detaches) all nodes starting from this node that
        fulfil the given condition. If any removed node is its parents
        only child, the parent is also removed.

        Arguments:
            condition: A function `f` where `f(node) == True` if `node`
                should be removed from the tree.
            include_starting_node: Whether to include the starting node
                or not. If False, the starting node will not be
                detached from its parent even though it fulfils the
                given condition. Defaults to True.

        Example: To remove all nodes at depth 2, use
            node.prune(lambda node: node.depth == 2)
        """
        nodes = self.traverse(filter=condition)
        for node in nodes:
            if not include_starting_node and node == self:
                continue

            if node.parent:
                if len(node.parent.children) == 1:
                    node.parent.detach()

            node.detach()

        if nodes:
            logger.info("Removed %d nodes from the tree %s", len(nodes), self.label)

    def max_depth(self) -> int:
        """Return the max depth of the tree starting at this node"""
        return max(node.depth for node in self.leaves())

    def is_root(self) -> bool:
        """True if this node is a root node"""
        return self.parent is None
