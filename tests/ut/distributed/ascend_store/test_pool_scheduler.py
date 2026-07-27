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

import os
import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    LookupKeyClient,
    get_zmq_rpc_path_lookup,
)


class TestGetZmqRpcPathLookup(unittest.TestCase):
    def test_default_port(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_0", result)
        self.assertIn("dp_rank0", result)

    def test_lookup_rpc_port(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 1
        config.kv_transfer_config.kv_connector_extra_config = {"lookup_rpc_port": 5555}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_5555", result)
        self.assertIn("dp_rank1", result)

    def test_mooncake_rpc_port_fallback(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {"mooncake_rpc_port": 6666}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_6666", result)


class TestKVPoolScheduler(unittest.TestCase):
    def _make_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.cache_config.block_size = block_size
        return config

    @staticmethod
    def _make_request(prompt_len, request_id="r1"):
        request = MagicMock()
        request.prompt_token_ids = list(range(prompt_len))
        request.num_tokens = prompt_len
        request.request_id = request_id
        request.block_hashes = [bytes([idx % 251]) for idx in range(max(prompt_len // 16, 1))]
        return request

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_consumer_no_load(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_too_short(self, mock_client_cls):
        config = self._make_config(block_size=64)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(32))
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_retention_threshold_below_skips_lookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = 16
        request = self._make_request(31)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        mock_client_cls.return_value.lookup.assert_not_called()
        self.assertNotIn("r1", scheduler.load_specs)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_retention_threshold_equal_runs_lookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = 16
        mock_client_cls.return_value.lookup.return_value = 16
        request = self._make_request(32)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 4), (12, False))
        mock_client_cls.return_value.lookup.assert_called_once_with(
            32,
            request.block_hashes,
            [0],
            hbm_hit_tokens=4,
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_retention_lookup_factor_changes_threshold(self, mock_client_cls):
        with patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RETENTION_LOOKUP_FACTOR": "3"}):
            scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = 16
        request = self._make_request(32)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        mock_client_cls.return_value.lookup.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_retention_threshold_above_runs_lookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = 16
        mock_client_cls.return_value.lookup.return_value = 32
        request = self._make_request(48)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (32, False))
        mock_client_cls.return_value.lookup.assert_called_once_with(
            48,
            request.block_hashes,
            [0],
            hbm_hit_tokens=0,
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_retention_threshold_below_still_builds_put_metadata(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = 16
        request = self._make_request(31)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        scheduler.update_state_after_alloc(request, MagicMock(), 0)

        scheduled_request = MagicMock()
        scheduled_request.req_id = "r1"
        scheduled_request.num_computed_tokens = 0
        scheduled_request.block_ids = [0]
        scheduled_request.prompt_token_ids = request.prompt_token_ids
        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.scheduled_new_reqs = [scheduled_request]
        scheduler_output.num_scheduled_tokens = {"r1": 16}
        scheduler_output.scheduled_cached_reqs.req_ids = []

        metadata = scheduler.build_connector_meta(scheduler_output)

        self.assertEqual(len(metadata.requests), 1)
        self.assertTrue(metadata.requests[0].can_save)
        self.assertIsNone(metadata.requests[0].load_spec)
        self.assertEqual(metadata.requests[0].num_prompt_tokens, 31)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_hbm_full_hit_skips_external_lookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        scheduler.retention_interval = None
        request = self._make_request(32)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 32), (0, False))
        mock_client_cls.return_value.lookup.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_hit(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, is_async = scheduler.get_num_new_matched_tokens(request, 16)
        self.assertEqual(need, 32)  # 48 - 16
        self.assertFalse(is_async)
        self.assertIn("r1", scheduler.load_specs)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_all_hit(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        # When external hit equals num_tokens, reduce by 1
        mock_client_cls.return_value.lookup.return_value = 64

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, _ = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(need, 63)
        self.assertEqual(scheduler.load_specs["r1"].kvpool_cached_tokens, 63)
        self.assertEqual(scheduler.load_specs["r1"].kvpool_store_skip_tokens, 64)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_less_than_computed(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 16

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, _ = scheduler.get_num_new_matched_tokens(request, 32)
        self.assertEqual(need, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_no_load_spec(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertIn("r1", scheduler._unfinished_request_ids)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_with_load(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 32

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        scheduler.get_num_new_matched_tokens(request, 0)
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 32)
        self.assertTrue(scheduler.load_specs["r1"].can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_zero_external(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import LoadSpec

        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=False)

        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertFalse(scheduler.load_specs["r1"].can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_consumer_no_put(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        result = scheduler.request_finished(request, [1, 2, 3])
        self.assertEqual(result, (False, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_no_tracker(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        # No tracker => tracker is None => num_saved_tokens check skipped
        # delay_free_blocks = len(block_ids) > 0 => True
        result = scheduler.request_finished(request, [1, 2])
        # tracker is None so condition `tracker.num_saved_tokens <= 0` is not checked
        # but tracker is None means `tracker is not None` is False => (False, None)
        # Actually: tracker = self._request_trackers.get("r1") => None
        # `if tracker is not None and tracker.num_saved_tokens <= 0:` => False
        # delay_free_blocks = len([1,2]) > 0 => True
        self.assertEqual(result, (True, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_with_saved_tokens(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [1, 2])
        self.assertTrue(delay)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_empty_blocks(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [])
        self.assertFalse(delay)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_async(self, mock_client_cls):
        config = self._make_config(extra_config={"load_async": True})
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(need, 48)
        self.assertTrue(is_async)


class TestKVPoolSchedulerBuildMeta(unittest.TestCase):
    def _make_config(self, kv_role="kv_producer", block_size=16):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.cache_config.block_size = block_size
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_new_req(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        # Setup a request via update_state_after_alloc
        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.num_computed_tokens = 0
        request.block_hashes = [b"h0", b"h1"]
        request.all_token_ids = list(range(32))
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 0)

        # Create scheduler output
        new_req_data = MagicMock()
        new_req_data.req_id = "r1"
        new_req_data.num_computed_tokens = 0
        new_req_data.block_ids = [0, 1]
        new_req_data.prompt_token_ids = list(range(32))

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = [new_req_data]
        sched_output.num_scheduled_tokens = {"r1": 32}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        meta = scheduler.build_connector_meta(sched_output)
        self.assertTrue(len(meta.requests) >= 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_finished_req(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [0, 1])
        scheduler._unfinished_request_ids.add("r1")

        sched_output = MagicMock()
        sched_output.finished_req_ids = {"r1"}
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = []
        sched_output.num_scheduled_tokens = {}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        self.assertNotIn("r1", scheduler._request_trackers)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_consumer_skip_save(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)

        new_req_data = MagicMock()
        new_req_data.req_id = "r1"
        new_req_data.num_computed_tokens = 0
        new_req_data.block_ids = [0, 1]
        new_req_data.prompt_token_ids = list(range(32))

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = [new_req_data]
        sched_output.num_scheduled_tokens = {"r1": 32}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        # Consumer with no consumer_is_to_put => force_skip_save

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_preempted(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [0, 1])

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = {"r1"}
        sched_output.scheduled_new_reqs = []
        sched_output.num_scheduled_tokens = {}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        self.assertNotIn("r1", scheduler._request_trackers)


class TestLookupKeyClient(unittest.TestCase):
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup(self, mock_zmq, mock_make_socket):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        mock_socket.recv.return_value = (32).to_bytes(4, "big")

        client = LookupKeyClient(config)
        client.encoder.encode.side_effect = [[b"hash-frame"], [b"group-frame"]]
        result = client.lookup(64, [b"\xaa\xbb"], [3], hbm_hit_tokens=16)
        self.assertEqual(result, 32)
        mock_socket.send_multipart.assert_called_once_with(
            [
                (64).to_bytes(4, "big"),
                b"group-frame",
                (16).to_bytes(4, "big"),
                b"hash-frame",
            ],
            copy=False,
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_close(self, mock_zmq, mock_make_socket):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket

        client = LookupKeyClient(config)
        client.close()
        mock_socket.close.assert_called_once_with(linger=0)


if __name__ == "__main__":
    unittest.main()
