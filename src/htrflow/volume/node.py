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

        self._id = f"node{id(self)}"  # A unique ID to fall back on if labels are not set
        self._local_label = label  # A local label, may not be unique within the tree
        self._global_label: str | None = None  # A global label created by chaining the ancestors' local labels

    @property
    def label(self) -> str:
        """The node's label. May be altered with node.relabel()"""
        return self._global_label or self._local_label or self._id

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

    def relabel(
        self,
        label_func: Callable[["Node"], str] = lambda _: "node",
        template: str = "{label}{number}",
        sep: str = "_",
        prefix: str = "",
        _counter=None,
    ) -> None:
        """Relabel the children of this node

        Assigns unique labels to the nodes of the tree. The nodes gets
        their label by appending to their parent's label. This
        overwrites the existing label, which is set to "node" at
        initialization.

        (Example) By calling node.relabel() with the default arguments
        on the root node, the tree would be relabeled as:

        root                         root
         ├── node                     ├── root_node0
         |    └── node                |    └── root_node0_node0
         └── node          ->         └── root_node1
              ├── node                     ├── root_node1_node0
              └── node                     └── root_node1_node1

        Note that this function does not change the label of the root
        node.

        Arguments:
            label_func: A function `f(node) -> str` that returns a
                descriptive label of `node`.
            template: A template for each node's appended section. It
                must include formatting placeholders for `label` and
                `number`, where `label` is placeholder for the label
                returned by `label_func` and `number` is a serial number
                assigned to the node. Defaults to "{label}{number}",
                which yields sections like "node2" or "line3". See
                str.format() for the formatting syntax.
            sep: A separator character that is inserted between each
                entry, defaults to "_".
            prefix: A prefix that is added to all labels.
            _counter: An argument used in recursive calls to this
                function. Should always be None elsewise. It keeps
                track of which labels exist and what numbers the next
                labels should get.
        """
        if _counter is None:
            _counter = defaultdict(lambda: count(0))
            if self._local_label:
                prefix += self._local_label

        long_template = f"{prefix}{sep}{template}".strip(sep)
        for child in self:
            label = label_func(child)
            unnumbered_label = long_template.format(label=label, number="X")
            number = next(_counter[unnumbered_label])
            child._local_label = template.format(label=label, number=number)
            child._global_label = long_template.format(label=label, number=number)
            child.relabel(label_func, template, sep, child._global_label, _counter)

    def relabel_levels(self, level_labels: list[str] | None = None, default: str = "node", **kwargs) -> None:
        """Relabel nodes level-by-level

        A simple way to assign labels whenever all nodes at the same
        depth (same level) share the same descriptive label. For
        example, if all nodes on the first level correspond to regions,
        and all nodes on the second level correspond to lines, passing
        level_labels = ["region", "line"] would yield:

        node                         node
         ├── node                     ├── node_region0
         |    └── node                |    └── node_region0_line0
         └── node          ->         └── region1
              ├── node                     ├── node_region1_line0
              └── node                     └── node_region1_line1

        Arguments:
            level_labels: A list of strings where the i:th element
                describes the nodes at the i:th level of the tree.
            default: A default label to assign whenever the tree is
                deeper than the number of passed level_labels. Defaults
                to "node".
            **kwargs: Optional formatting keyword arguments that are
                forwarded to node.relabel(). See node.relabel() for
                details.
        """
        if level_labels is None:
            self.relabel(lambda _: default)
            return

        def label_func(node: "Node") -> str:
            depth = node.depth()
            if depth > len(level_labels):
                return default
            return level_labels[depth - 1]

        self.relabel(label_func, **kwargs)

    def depth(self) -> int:
        """Depth of node

        The number of edges (or "generations") between this node and
        the root node. The depth of the root node is 0. The depth of
        its children is 1, and so on.
        """
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

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
            node.prune(lambda node: node.depth() == 2)
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
        return max(node.depth() for node in self.leaves())

    def is_root(self) -> bool:
        """True if this node is a root node"""
        return self.parent is None
