#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.exists_calls = []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        self.exists_calls.append(list(keys))
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class FakeKey:
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return self._val


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.key_build_starts = []
        self.key_build_groups = []

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, h in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num:
                continue
            yield start, end, PoolKey(meta, f"k{i}")

    def process_token_key_strings_with_block_ids(
        self,
        token_len,
        block_hashes,
        block_ids,
        mask_num=0,
        kv_cache_group_id=0,
        skip_null_blocks=False,
        cache_role="kv",
        chunk_filter=None,
    ):
        del cache_role
        self.key_build_groups.append(kv_cache_group_id)
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, block_hash in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len or i >= len(block_ids):
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num or (skip_null_blocks and block_ids[i] <= 0):
                continue
            if chunk_filter is not None and not chunk_filter(start):
                continue
            self.key_build_starts.append(start)
            yield start, end, PoolKey(meta, f"k{i}").to_string(), block_hash, block_ids[i]

    def can_use_sparse_store_mask_key_build(self, *args, **kwargs):
        return False

    def _get_store_granularity(self, _kv_cache_group_id=0):
        return self.block_size

    def prepare_value(self, start, end, block_ids, block_id=None, **kwargs):
        del kwargs
        if block_id is None:
            block_id = block_ids[start // self.block_size]
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start]

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class MaskedFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self, block_size=16, store_mask=None, load_mask=None):
        super().__init__(block_size)
        self._store_mask = store_mask
        self._load_mask = load_mask
        self.store_mask_calls = []
        self.load_mask_calls = []

    def store_mask(self, token_len, num_prompt_tokens=None):
        self.store_mask_calls.append((token_len, num_prompt_tokens))
        return self._store_mask

    def load_mask(self, block_hashes, token_len):
        self.load_mask_calls.append((block_hashes, token_len))
        return self._load_mask


class SparseFakeTokenDatabase(MaskedFakeTokenDatabase):
    def __init__(self):
        super().__init__(block_size=4, store_mask=([True, False, True, True],))
        self.sparse_calls = []

    def can_use_sparse_store_mask_key_build(self, *args, **kwargs):
        return True

    def process_token_key_strings_with_block_ids(self, *args, **kwargs):
        raise AssertionError("generic key builder must not run on sparse fast path")

    def process_token_key_strings_with_block_ids_sparse_store_mask(
        self,
        token_len,
        block_hashes,
        block_ids,
        store_mask,
        kv_cache_group_id=0,
        skip_null_blocks=False,
        shard_rank=None,
        shard_size=None,
    ):
        self.sparse_calls.append((list(store_mask), shard_rank, shard_size))
        candidate_index = 0
        for i, allowed in enumerate(store_mask):
            if not allowed or i >= len(block_ids):
                continue
            if skip_null_blocks and block_ids[i] <= 0:
                continue
            if shard_size and shard_size > 1:
                selected = candidate_index % shard_size == shard_rank
                candidate_index += 1
                if not selected:
                    continue
            start = i * self.block_size
            end = min(start + self.block_size, token_len)
            yield start, end, f"sparse-k{i}", block_hashes[i], block_ids[i]


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        return t, store

    def test_add_request(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

    def test_get_and_clear_finished_requests(self):
        t, _ = self._make_thread()
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        finished = t.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup_all_exist(self):
        t, _ = self._make_thread([1, 1, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, True, True])

    def test_lookup_partial(self):
        t, _ = self._make_thread([1, 0, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_exception(self):
        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        result = t.lookup(["k1"])
        self.assertEqual(result, [False])

    def test_update_and_get_kv_events(self):
        t, _ = self._make_thread()
        event1 = BlockStored(block_hashes=["h1"])
        event2 = BlockStored(block_hashes=["h2"])
        t.update_kv_event([event1, event2])
        events = t.get_kv_events()
        self.assertEqual(len(events), 2)
        # After get, events should be cleared
        self.assertEqual(len(t.get_kv_events()), 0)

    def test_handle_request_base_noop(self):
        t, _ = self._make_thread()
        # Base class _handle_request does nothing
        t._handle_request(MagicMock())

    def test_process_request_balances_queue_on_success(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.request_queue.put(req)

        t._process_request(req)

        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_process_request_balances_queue_and_calls_exception_hook(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.request_queue.put(req)
        t._handle_request = MagicMock(side_effect=RuntimeError("boom"))
        t._handle_request_exception = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            t._process_request(req)

        t._handle_request_exception.assert_called_once_with(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)

    def test_handle_request_applies_store_mask_before_exists(self):
        store = FakeStore([0, 0])
        db = MaskedFakeTokenDatabase(store_mask=([False, True, False, True],))
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            num_prompt_tokens=64,
        )
        t.add_stored_request("r1")

        t._handle_request(req)

        self.assertEqual(db.store_mask_calls, [(64, 64)])
        self.assertEqual(len(store.exists_calls[0]), 2)
        self.assertTrue(store.exists_calls[0][0].endswith("@k1"))
        self.assertTrue(store.exists_calls[0][1].endswith("@k3"))

    def test_full_kvpool_hit_skips_exists_and_put(self):
        store = FakeStore([0, 0, 0, 0])
        db = FakeTokenDatabase(block_size=4)
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
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
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=15,
                kvpool_store_skip_tokens=16,
                can_load=False,
            ),
        )

        t._handle_stored_request(req)

        self.assertEqual(db.key_build_starts, [])
        self.assertEqual(store.exists_calls, [])
        self.assertEqual(store.put_calls, [])

    def test_hbm_hit_still_checks_exists_while_kvpool_hit_skips(self):
        store = FakeStore([0])
        db = FakeTokenDatabase(block_size=4)
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=4,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="mixed-hit",
            token_len_chunk=16,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            load_spec=LoadSpec(
                vllm_cached_tokens=4,
                kvpool_cached_tokens=15,
                kvpool_store_skip_tokens=16,
                can_load=False,
            ),
        )

        t._handle_stored_request(req)

        self.assertEqual(db.key_build_starts, [0])
        self.assertEqual(len(store.exists_calls), 1)
        self.assertEqual(len(store.exists_calls[0]), 1)
        self.assertEqual(len(store.put_calls), 1)

    def test_sparse_pre_shard_builds_only_local_candidates(self):
        store = FakeStore([0])
        db = SparseFakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=4,
            tp_rank=1,
            dcp_size=1,
            put_step=2,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        t._put_sparse_store_mask = True
        t._put_pre_shard_key_build = True
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[10, 11, 12, 13],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            num_prompt_tokens=16,
        )

        t._handle_stored_request(req)

        self.assertEqual(db.sparse_calls, [([True, False, True, True], 1, 2)])
        self.assertEqual(store.exists_calls, [["sparse-k2"]])
        self.assertEqual(store.put_calls[0][0], ["sparse-k2"])

    def test_put_lookup_checks_each_group_without_c128_gate(self):
        store = FakeStore([0])
        db = FakeTokenDatabase(block_size=16)
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=[16, 16],
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids_by_group=[[0], [10]],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0, 1],
            kv_cache_families_by_group=["c128", "c4"],
        )

        t._handle_stored_request(req)

        self.assertEqual(db.key_build_groups, [0, 1])
        self.assertEqual(len(store.exists_calls), 2)
        self.assertEqual(len(store.put_calls), 2)

    def test_handle_request_all_exist_no_put(self):
        t, store = self._make_thread([1, 1])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_in_stored(self):
        t, store = self._make_thread([0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_consumer_role(self):
        t, store = self._make_thread([0], kv_role="kv_consumer")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_dec_nonexistent_request(self):
        t, _ = self._make_thread()
        t.dec_stored_request("nonexist")  # should not raise

    def test_delete_nonexistent_request(self):
        t, _ = self._make_thread()
        t.delete_finished_stored_request("nonexist")  # should not raise

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=event,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore([0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_applies_load_mask_before_get(self):
        store = FakeStore()
        db = MaskedFakeTokenDatabase(load_mask=([False, True],))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )

        t._handle_request(req)

        self.assertEqual(db.load_mask_calls, [([b"h0", b"h1"], 32)])
        self.assertEqual(len(store.get_calls[0][0]), 1)
        self.assertTrue(store.get_calls[0][0][0].endswith("@k1"))

    def test_process_request_marks_request_finished_on_get_error(self):
        store = FakeStore()
        store.get = MagicMock(side_effect=RuntimeError("get failed"))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            load_spec=LoadSpec(0, 16, can_load=True, token_len=16),
        )
        t.request_queue.put(req)

        with self.assertRaisesRegex(RuntimeError, "get failed"):
            t._process_request(req)

        self.assertIn("r1", t.get_and_clear_finished_requests())
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        _meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=event,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


if __name__ == "__main__":
    unittest.main()
