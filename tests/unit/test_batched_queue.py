from htrflow.pipeline.batched_queue import BatchedQueue


def test_batched_queue_timeout():
    item = 1

    # We ask of batch size 100, but only put one item in it
    queue = BatchedQueue(100)
    queue.put(item)

    # The queue should release an incomplete batch (i.e., not raise queue.Empty here)
    queue.get(timeout=1)


def test_batched_queue_batching():
    n_items = 10
    batch_size = 2
    items = list(range(n_items))

    q = BatchedQueue(batch_size)
    for item in items:
        q.put(item)

    batches = []
    while 1:
        batch = q.get(timeout=0.1)
        if not batch:
            break
        batches.append(batch)

    assert len(batches) == n_items // batch_size


def test_batched_queue_return_original_items():

    q = BatchedQueue(2)

    n_items = 10
    items = list(range(n_items))
    for item in items:
        q.put(item)

    batches = []
    while 1:
        batch = q.get(timeout=0.1)
        if not batch:
            break
        batches.append(batch)

    assert {item.item for batch in batches for item in batch} == set(items)
