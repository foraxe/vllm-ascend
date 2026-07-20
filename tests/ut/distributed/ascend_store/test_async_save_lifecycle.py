import importlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PREFIX = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
SOURCE_DIR = REPO_ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"


def load_local_modules():
    package = importlib.import_module(MODULE_PREFIX)
    source_dir = str(SOURCE_DIR)
    if source_dir not in package.__path__:
        package.__path__.insert(0, source_dir)
    for leaf_name in (
        "pool_worker",
        "kv_transfer",
        "coordinator",
        "config_data",
        "kvtrace",
        "pool_scheduler",
    ):
        sys.modules.pop(f"{MODULE_PREFIX}.{leaf_name}", None)
    config_data = importlib.import_module(f"{MODULE_PREFIX}.config_data")
    kv_transfer = importlib.import_module(f"{MODULE_PREFIX}.kv_transfer")
    pool_worker = importlib.import_module(f"{MODULE_PREFIX}.pool_worker")
    pool_scheduler = importlib.import_module(f"{MODULE_PREFIX}.pool_scheduler")
    return config_data, kv_transfer, pool_worker, pool_scheduler


local_config_data, local_kv_transfer, local_pool_worker, local_pool_scheduler = load_local_modules()
AscendConnectorMetadata = local_config_data.AscendConnectorMetadata
LoadSpec = local_config_data.LoadSpec
ReqMeta = local_config_data.ReqMeta
KVCacheStoreSendingThread = local_kv_transfer.KVCacheStoreSendingThread
KVCacheStoreRecvingThread = local_kv_transfer.KVCacheStoreRecvingThread
KVCacheStoreLayerRecvingThread = local_kv_transfer.KVCacheStoreLayerRecvingThread
KVPoolScheduler = local_pool_scheduler.KVPoolScheduler
KVPoolWorker = local_pool_worker.KVPoolWorker


@pytest.fixture(autouse=True)
def stable_put_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PUT_KEY_STRING_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_ASCEND_PUT_SPARSE_STORE_MASK", "0")
    monkeypatch.setenv("VLLM_ASCEND_PUT_PRE_SHARD_KEY_BUILD", "0")


class FakeStore:
    def __init__(self):
        self.exists_calls = []
        self.put_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        self.exists_calls.append(list(keys))
        return [0] * len(keys)

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


_DEFAULT_STORE_MASK = object()


class FakeTokenDatabase:
    def __init__(self, block_size=4, store_mask=_DEFAULT_STORE_MASK):
        self.block_size = block_size
        self._store_mask = (
            [True, True, True, True]
            if store_mask is _DEFAULT_STORE_MASK
            else store_mask
        )
        self.key_build_starts = []

    def store_mask(self, _token_len, _num_prompt_tokens):
        if self._store_mask is None:
            return None
        return (list(self._store_mask),)

    def can_use_sparse_store_mask_key_build(self, *args, **kwargs):
        return False

    def _get_store_granularity(self, _kv_cache_group_id=0):
        return self.block_size

    def process_token_key_strings_with_block_ids(
        self,
        token_len,
        block_hashes,
        block_ids,
        kv_cache_group_id=0,
        skip_null_blocks=False,
        chunk_filter=None,
    ):
        del kv_cache_group_id, skip_null_blocks
        for index, start in enumerate(range(0, token_len, self.block_size)):
            end = min(start + self.block_size, token_len)
            if index >= len(block_ids):
                break
            if chunk_filter is not None and not chunk_filter(start):
                continue
            self.key_build_starts.append(start)
            chunk_hash = block_hashes[index] if index < len(block_hashes) else b""
            yield start, end, f"k{start}-{end}", chunk_hash, block_ids[index]

    def prepare_value(self, start, end, block_ids, **kwargs):
        block_id = kwargs.get("block_id")
        if block_id is None:
            block_id = block_ids[start // self.block_size]
        return [block_id], [end - start], None


def make_req(req_id="r1"):
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=16,
        block_ids=[0],
        block_hashes=[b"h0"],
        can_save=True,
    )


def make_worker(send_thread):
    worker = object.__new__(KVPoolWorker)
    worker.group_uses_align_state = [False]
    worker.kv_send_thread = send_thread
    return worker


def make_metadata(req):
    meta = AscendConnectorMetadata(set(), set())
    meta.add_request(req)
    return meta


def make_scheduler_config(block_size=4):
    kv_transfer_config = SimpleNamespace(
        kv_role="kv_producer",
        kv_connector_extra_config={},
        get_from_extra_config=lambda _name, default: default,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(),
            hf_config=SimpleNamespace(),
        ),
        kv_transfer_config=kv_transfer_config,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            hash_block_size=None,
        ),
    )


def test_scheduler_preserves_raw_kvpool_hit_for_put_skip(monkeypatch):
    fake_client = MagicMock()
    fake_client.lookup.return_value = 16
    monkeypatch.setattr(local_pool_scheduler, "LookupKeyClient", lambda _config: fake_client)
    scheduler = KVPoolScheduler(make_scheduler_config(), use_layerwise=False)
    request = SimpleNamespace(
        request_id="full-hit",
        prompt_token_ids=list(range(16)),
        num_tokens=16,
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (15, False)
    load_spec = scheduler.load_specs["full-hit"]
    assert load_spec.kvpool_cached_tokens == 15
    assert load_spec.kvpool_store_skip_tokens == 16


def test_async_save_switch_returns_without_queue_join(monkeypatch):
    monkeypatch.setattr("torch.npu.Event", MagicMock(return_value=MagicMock()))
    send_thread = MagicMock()
    send_thread._per_request_save_wait = False
    send_thread._async_save = True
    send_thread.request_queue = MagicMock()
    send_thread.request_queue.qsize.return_value = 1
    send_thread.request_queue.join.side_effect = AssertionError("async save must not join")
    send_thread.stored_requests = {"r1": 1}
    worker = make_worker(send_thread)
    req = make_req()

    worker.wait_for_save(make_metadata(req))

    send_thread.add_stored_request.assert_called_once_with("r1")
    send_thread.add_request.assert_called_once_with(req)
    send_thread.request_queue.join.assert_not_called()


def test_per_request_wait_takes_precedence_over_async_switch(monkeypatch):
    monkeypatch.setattr("torch.npu.Event", MagicMock(return_value=MagicMock()))
    event = MagicMock()
    event.wait.return_value = True
    send_thread = MagicMock()
    send_thread._per_request_save_wait = True
    send_thread._async_save = True
    send_thread.prepare_stored_request_done_event.return_value = event
    send_thread.request_queue = MagicMock()
    send_thread.request_queue.qsize.return_value = 0
    send_thread.request_queue.join.side_effect = AssertionError("per-request wait must not global join")
    send_thread.stored_requests = {"r1": 1}
    worker = make_worker(send_thread)
    req = make_req()

    worker.wait_for_save(make_metadata(req))

    send_thread.prepare_stored_request_done_event.assert_called_once_with("r1")
    event.wait.assert_called_once_with(timeout=300)
    send_thread.request_queue.join.assert_not_called()


def test_process_request_exception_dec_stored_request_and_task_done():
    thread = KVCacheStoreSendingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = make_req()
    thread.add_stored_request("r1")
    thread.request_queue.put(req)
    thread._handle_request = MagicMock(side_effect=RuntimeError("save failed"))

    with pytest.raises(RuntimeError, match="save failed"):
        thread._process_request(req)

    assert thread.stored_requests["r1"] == 0
    assert thread.request_queue.unfinished_tasks == 0


def test_process_request_early_return_marks_task_done():
    thread = KVCacheStoreSendingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = make_req()
    thread.request_queue.put(req)

    thread._process_request(req)

    assert thread.request_queue.unfinished_tasks == 0


def test_process_recv_exception_marks_finished_and_task_done():
    thread = KVCacheStoreRecvingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        ready_event=threading.Event(),
    )
    req = make_req()
    thread.request_queue.put(req)
    thread._handle_request = MagicMock(side_effect=RuntimeError("load failed"))

    with pytest.raises(RuntimeError, match="load failed"):
        thread._process_request(req)

    assert thread.get_and_clear_finished_requests() == {"r1"}
    assert thread.request_queue.unfinished_tasks == 0


def test_process_layer_recv_exception_sets_event_and_task_done():
    get_event = threading.Event()
    thread = KVCacheStoreLayerRecvingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        ready_event=threading.Event(),
        get_event=get_event,
    )
    req = SimpleNamespace(req_id="r1")
    thread.request_queue.put(req)
    thread._handle_request = MagicMock(side_effect=RuntimeError("layer load failed"))

    with pytest.raises(RuntimeError, match="layer load failed"):
        thread._process_request(req)

    assert get_event.is_set()
    assert thread.request_queue.unfinished_tasks == 0


def test_get_finished_does_not_release_while_save_pending():
    worker = object.__new__(KVPoolWorker)
    worker.kv_role = "kv_producer"
    worker.consumer_is_to_put = False
    worker.load_async = False
    worker.finished_store_req = set()
    send_thread = SimpleNamespace(
        stored_requests={"r1": 1},
        get_and_clear_finished_requests=lambda: set(),
        delete_finished_stored_request=MagicMock(),
    )
    worker.kv_send_thread = send_thread

    result = worker.get_and_clear_finished_requests(
        {"r1"}, AscendConnectorMetadata(set(), set())
    )

    assert result == set()
    assert "r1" in worker.finished_store_req


def test_full_kvpool_hit_minus_one_uses_raw_hit_to_skip_tail_chunk():
    thread = KVCacheStoreSendingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="full-hit",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=15,
            can_load=False,
            kvpool_store_skip_tokens=16,
        ),
    )

    skip_range = thread._kvpool_hit_skip_range(req)
    adjusted_mask, skipped = thread._apply_kvpool_hit_skip_to_store_mask(
        [True, True, True, True],
        0,
        skip_range,
    )

    assert skip_range == (0, 16)
    assert adjusted_mask == [False, False, False, False]
    assert skipped == 4
    assert thread._should_skip_kvpool_hit_chunk(12, 16, skip_range)


def test_partial_kvpool_hit_without_store_skip_keeps_tail_chunk():
    thread = KVCacheStoreSendingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="partial-hit",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=15,
            can_load=False,
        ),
    )

    skip_range = thread._kvpool_hit_skip_range(req)
    adjusted_mask, skipped = thread._apply_kvpool_hit_skip_to_store_mask(
        [True, True, True, True],
        0,
        skip_range,
    )

    assert skip_range == (0, 15)
    assert adjusted_mask == [False, False, False, True]
    assert skipped == 3
    assert not thread._should_skip_kvpool_hit_chunk(12, 16, skip_range)


def test_hbm_cached_chunk_keeps_exists_while_kvpool_hit_chunks_skip():
    thread = KVCacheStoreSendingThread(
        m_store=FakeStore(),
        token_database=FakeTokenDatabase(),
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="hbm-plus-kvpool-hit",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=4,
            kvpool_cached_tokens=15,
            can_load=False,
            kvpool_store_skip_tokens=16,
        ),
    )

    skip_range = thread._kvpool_hit_skip_range(req)
    adjusted_mask, skipped = thread._apply_kvpool_hit_skip_to_store_mask(
        [True, True, True, True],
        0,
        skip_range,
    )

    assert skip_range == (4, 16)
    assert adjusted_mask == [True, False, False, False]
    assert skipped == 3
    assert not thread._should_skip_kvpool_hit_chunk(0, 4, skip_range)
    assert thread._should_skip_kvpool_hit_chunk(4, 8, skip_range)


def test_handle_stored_request_applies_full_kvpool_hit_skip_before_exists():
    store = FakeStore()
    token_database = FakeTokenDatabase()
    thread = KVCacheStoreSendingThread(
        m_store=store,
        token_database=token_database,
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="full-hit-main-path",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=15,
            can_load=False,
            kvpool_store_skip_tokens=16,
        ),
    )

    thread._handle_stored_request(req)

    assert token_database.key_build_starts == []
    assert store.exists_calls == []
    assert store.put_calls == []


def test_handle_stored_request_skips_kvpool_hits_without_store_mask():
    store = FakeStore()
    token_database = FakeTokenDatabase(store_mask=None)
    thread = KVCacheStoreSendingThread(
        m_store=store,
        token_database=token_database,
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="hbm-plus-kvpool-no-store-mask",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=4,
            kvpool_cached_tokens=15,
            can_load=False,
            kvpool_store_skip_tokens=16,
        ),
    )

    thread._handle_stored_request(req)

    assert token_database.key_build_starts == [0]
    assert store.exists_calls == [["k0-4"]]
    assert store.put_calls[0][0] == ["k0-4"]


def test_handle_stored_request_keeps_hbm_chunk_exists_and_skips_kvpool_hits():
    store = FakeStore()
    token_database = FakeTokenDatabase()
    thread = KVCacheStoreSendingThread(
        m_store=store,
        token_database=token_database,
        block_size=4,
        tp_rank=0,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    req = ReqMeta(
        req_id="hbm-plus-kvpool-main-path",
        token_len_chunk=16,
        block_ids=[0, 1, 2, 3],
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        can_save=True,
        load_spec=LoadSpec(
            vllm_cached_tokens=4,
            kvpool_cached_tokens=15,
            can_load=False,
            kvpool_store_skip_tokens=16,
        ),
    )

    thread._handle_stored_request(req)

    assert token_database.key_build_starts == [0]
    assert store.exists_calls == [["k0-4"]]
    assert store.put_calls[0][0] == ["k0-4"]
