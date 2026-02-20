import queue

from htrflow.pipeline.batched_queue import BatchedQueue


def test_batched_queue_patience():
    item = 1
    patience = 0.1

    # We ask of batch size 100, but only put one item in it
    queue = BatchedQueue(100, patience=patience)
    queue.put(item)

    # The queue should release an incomplete batch (i.e., not raise
    # queue.Empty here)
    queue.get(timeout=patience * 10)


def test_batched_queue_batching():
    n_items = 10
    batch_size = 2
    items = list(range(n_items))

    q = BatchedQueue(batch_size, patience=0.1)
    for item in items:
        q.put(item)

    batches = []
    while 1:
        try:
            batch = q.get(timeout=0.1)
            batches.append(batch)
        except queue.Empty:
            break

    assert len(batches) == n_items // batch_size


def test_batched_queue_return_original_items():

    q = BatchedQueue(2, patience=0.1)

    n_items = 10
    items = list(range(n_items))
    for item in items:
        q.put(item)

    batches = []
    while 1:
        try:
            batch = q.get(timeout=0.1)
            batches.append(batch)
        except queue.Empty:
            break

    assert {item.item for batch in batches for item in batch} == set(items)
