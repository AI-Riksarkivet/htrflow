import logging
import queue
import threading
from concurrent.futures import Future


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

    def __init__(self, batch_size, patience=0.1):
        """
        Arguments:
            batch_size: Target batch size.
            patience: Time to wait for an incoming item before releasing
                an imcomplete batch.
        """

        self.batch_size = batch_size
        self.patience = patience

        self._in = queue.Queue()
        self._out = queue.Queue()
        self._thread = threading.Thread(target=self._process, daemon=True)
        self._thread.start()

    def _process(self):
        while 1:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    item = self._in.get(timeout=self.patience)
                    batch.append(item)
                except queue.Empty:
                    if batch:
                        logging.debug("Releasing an incomplete batch (size %d)", len(batch))
                        break
                    continue

            self._out.put(batch)

    def get(self, block: bool = True, timeout: float | None = None) -> list[ItemWithFuture]:
        return self._out.get(block, timeout)

    def put(self, item, block: bool = True, timeout: float | None = None):
        item = ItemWithFuture(item)
        self._in.put(item, block, timeout)
        return item.future
