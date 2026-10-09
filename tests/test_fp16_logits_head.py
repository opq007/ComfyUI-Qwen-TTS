"""Regression tests for the fp16-safe logits head.

The code predictor's MLP products legitimately exceed float16's ~6.5e4 range
in the bf16-trained checkpoints (measured ~1.4e5 on the 0.6B weights, upstream
QwenLM/Qwen3-TTS#43): under an fp16 model the predictor runs in fp32 (see
Qwen3TTSForConditionalGeneration.from_pretrained) and the logits heads compute
in float32 via ``_fp32_logits_head``. Without that, the head matmul output is
stored as fp16, overflows to +inf, and the NaN probabilities derail sampling
so the generated audio stops following the text.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from qwen_tts.core.models.modeling_qwen3_tts import _fp32_logits_head


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class Fp32LogitsHeadTests(unittest.TestCase):
    def _head(self, dtype, with_bias=False):
        torch.manual_seed(0)
        head = nn.Linear(64, 32, bias=with_bias).to(dtype)
        with torch.no_grad():
            head.weight.normal_(0.0, 300.0)  # true logits land far above 65504
        return head

    def test_fp16_head_output_overflow_is_prevented(self):
        # hidden ~O(300) x weights ~O(300) over 64 dims -> logits ~O(1e5):
        # the raw fp16 head stores +inf where the true value exceeds 65504.
        torch.manual_seed(1)
        head = self._head(torch.float16)
        hidden = (torch.randn(2, 5, 64) * 300).to(torch.float16)

        with torch.no_grad():
            raw = head(hidden)
            fixed = _fp32_logits_head(head, hidden)
            truth = nn.functional.linear(hidden.float(), head.weight.float(), head.bias)

        self.assertTrue(torch.isinf(raw).any(), "raw fp16 head should overflow in this scenario")
        self.assertFalse(torch.isinf(fixed).any())
        self.assertFalse(torch.isnan(fixed).any())
        self.assertEqual(fixed.dtype, torch.float32)
        self.assertTrue(torch.allclose(fixed, truth, rtol=1e-5, atol=1e-3))

    def test_fp32_and_bf16_inputs_pass_through(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                head = self._head(dtype, with_bias=True)
                hidden = torch.randn(2, 5, 64).to(dtype)
                with torch.no_grad():
                    out = _fp32_logits_head(head, hidden)
                self.assertEqual(out.dtype, dtype)
                expected = nn.functional.linear(hidden, head.weight, head.bias)
                self.assertTrue(torch.equal(out, expected))

    def test_fp16_no_bias(self):
        head = self._head(torch.float16, with_bias=False)
        hidden = (torch.randn(1, 3, 64) * 300).to(torch.float16)
        with torch.no_grad():
            out = _fp32_logits_head(head, hidden)
        self.assertFalse(torch.isinf(out).any())
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
