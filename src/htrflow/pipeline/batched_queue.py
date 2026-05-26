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

    def __init__(self, batch_size):
        """
        Arguments:
            batch_size: Target batch size.
        """

        self.batch_size = batch_size

        self._queue = queue.Queue()

    def get(self, block: bool = True, timeout: float | None = 0.1) -> list[ItemWithFuture]:
        batch = []
        while len(batch) < self.batch_size:
            try:
                item = self._queue.get(block=block, timeout=timeout)
                batch.append(item)
            except queue.Empty:
                if batch:
                    logger.debug("Releasing an incomplete batch (size %d of %d)", len(batch), self.batch_size)
                    break
        return batch

    def put(self, item, block: bool = True, timeout: float | None = None):
        item = ItemWithFuture(item)
        self._queue.put(item, block, timeout)
        return item.future
