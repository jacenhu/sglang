# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""End-to-end test for the /weights_checker endpoint with an MTP draft runner.

test_weight_checker_e2e.py covers the plain target-only path. This file proves
the draft-runner fan-out (selector in {draft, all}) is FUNCTIONAL on a real
partial model: an MTP/NEXTN draft shares only embed/head with the target and
carries its own private MTP layers, so a weight sync that misses those private
weights must be caught. miles' check_weight_update_equal relies on exactly this
(snapshot -> reset -> sync -> compare); without the fan-out a missed MTP sync is
invisible.

Model: MiMo-7B-RL -- the smallest real MTP-in-checkpoint model (shares embed/head
with the target via set_embed_and_head), so it exercises the same fan-out path as
the large NEXTN models on a single GPU.
"""

import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, suite="nightly-1-gpu", nightly=True)

_MODEL_NAME = "XiaomiMiMo/MiMo-7B-RL"
# The MTP draft shares only embed/head with the target (set_embed_and_head);
# every other draft.* weight is MTP-private -- precisely what a target-only check
# cannot see. Classify by name rather than a hardcoded count (which shifts with
# quant/fusion), so the test stays valid across MTP models.
_SHARED_WITH_TARGET = ("embed_tokens", "lm_head")


def _local_name(checksum_key: str) -> str:
    # Drop the runner-role prefix the scheduler adds to draft checksum keys
    # ("draft." or "draft_step_<i>."); compare() error names are already local.
    return checksum_key.split(".", 1)[1] if "." in checksum_key else checksum_key


def _is_mtp_private(local_name: str) -> bool:
    return not any(s in local_name for s in _SHARED_WITH_TARGET)


class TestWeightCheckerMTPE2E(CustomTestCase):
    """All cases share one launched server. test_z_* mutates weights to a sentinel
    and is named to sort last; the server is torn down right after, so leaving the
    engine corrupted is harmless."""

    @classmethod
    def setUpClass(cls):
        cls.url = DEFAULT_URL_FOR_TEST
        # mem-fraction 0.5 mirrors test/manual/models/test_mtp_models.py and leaves
        # ample free GPU for _compare's snapshot CPU->GPU round trip.
        cls.process = popen_launch_server(
            _MODEL_NAME,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, action: str, selector: str = None) -> requests.Response:
        payload = {"action": action}
        if selector is not None:
            payload["selector"] = selector
        return requests.post(
            f"{self.url}/weights_checker", json=payload, timeout=600
        )

    def _checksum_keys(self, selector: str) -> dict:
        r = self._post("checksum", selector)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["success"], body)
        keys = {}
        for rank in body["ranks"]:
            keys.update(rank.get("checksums", {}))
        return keys

    def test_a_fanout_covers_mtp_draft(self):
        """selector=all must fan out to the draft runner and add MTP-private
        weights on top of the target scope. If it added nothing, the checker
        would silently cover zero draft weights -- the dangerous vacuous pass."""
        target_keys = self._checksum_keys("target")
        all_keys = self._checksum_keys("all")
        self.assertTrue(target_keys, "target scope returned no weights")
        self.assertTrue(
            set(target_keys).issubset(set(all_keys)),
            "selector=all dropped target keys",
        )
        draft_keys = [k for k in all_keys if k.startswith("draft")]
        self.assertTrue(
            draft_keys, "selector=all added no draft.* keys; MTP fan-out not engaged"
        )
        private = [k for k in draft_keys if _is_mtp_private(_local_name(k))]
        self.assertTrue(
            private, f"no MTP-private weights under the draft runner: {draft_keys}"
        )

    def test_b_clean_compare_passes(self):
        """No false positive: a snapshot compared against unchanged weights passes."""
        self.assertTrue(self._post("snapshot", "all").json()["success"])
        r = self._post("compare", "all")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["success"], r.text)

    def test_z_poisoned_mtp_weight_is_caught(self):
        """The dirty case: poison the draft (MTP) weights, then the checker must
        catch it. compare(draft) fails and names MTP-private weights; compare(all)
        -- miles' actual call -- fails too, so a missed sync halts training."""
        self.assertTrue(self._post("snapshot", "all").json()["success"])
        self.assertTrue(self._post("reset_tensors", "draft").json()["success"])

        r = self._post("compare", "draft")
        body = r.json()
        self.assertEqual(r.status_code, 400, body)
        self.assertFalse(body["success"], body)
        names = set(re.findall(r"name=(\S+)", body["message"]))
        private = [n for n in names if _is_mtp_private(n)]
        self.assertTrue(
            private,
            f"compare(draft) failed but named no MTP-private weight: {sorted(names)}",
        )

        r_all = self._post("compare", "all")
        self.assertEqual(r_all.status_code, 400, r_all.text)
        self.assertFalse(r_all.json()["success"], r_all.text)


if __name__ == "__main__":
    unittest.main()
