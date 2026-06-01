import logging
import queue
from concurrent.futures import Future


logger = logging.getLogger(__name__)


class ItemWithFuture:
    """
    Wrapper class that gives an arbitrary item a Future.
    """

    def __init__(self, item):
        self.item = item
        self.future = Future()


class BatchedQueue:
    """
    Batched queue.

    A queue in which single items go in and comes out in batches of a given size:

        [a] [b] => BatchedQueue(2) => [a, b]
    """

    def __init__(self, batch_size: int):
        """
        Create a batched queue.

        Arguments:
            batch_size: Target batch size.
        """

        self.batch_size = batch_size
        self._queue: queue.Queue[ItemWithFuture] = queue.Queue()

    def get(self, timeout: float | None = 0.1) -> list[ItemWithFuture]:
        """
        Get a batch of items from the queue.

        Arguments:
            timeout: Maximum number of seconds to wait for a completing a batch.

        Returns:
            A list of at most `self.batch_size` items; possibly fewer or
            even zero items if no items were received within the given
            timeout.
        """
        batch: list[ItemWithFuture] = []
        while len(batch) < self.batch_size:
            try:
                item = self._queue.get(timeout=timeout)
                batch.append(item)
            except queue.Empty:
                logger.debug("Releasing an incomplete batch (size %d of %d)", len(batch), self.batch_size)
                break
        return batch

    def put(self, item):
        """
        Put a single item in the queue.

        Arguments:
            item: Item to put in queue.
        """
        item = ItemWithFuture(item)
        self._queue.put(item)
        return item.future
