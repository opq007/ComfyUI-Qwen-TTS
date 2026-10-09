"""Regression tests for the transformers>=5 ``create_causal_mask`` shim.

Issue (upstream #111): the shim's original guard ``if not getattr(create_causal_mask,
"_v5_shim", False)`` also fired on transformers 4.x, renaming the vendored
``input_embeds`` keyword into ``inputs_embeds`` — a name 4.57.x rejects with
``TypeError: create_causal_mask() got an unexpected keyword argument 'inputs_embeds'``.
The current guard probes the real signature instead.

No torch/transformers needed: the shim block is executed against stubs whose
signatures mirror real transformers releases (verified against PyPI wheels of
4.57.0/4.57.6, 5.0.0, 5.1.0, 5.5.0, 5.10.1, 5.15.0 and 5.19.0).
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELING_FILES = [
    ROOT / "qwen_tts" / "core" / "models" / "modeling_qwen3_tts.py",
    ROOT / "qwen_tts" / "core" / "tokenizer_12hz" / "modeling_qwen3_tts_tokenizer_v2.py",
]

PROBE_GUARD = 'if "inputs_embeds" in _inspect.signature(create_causal_mask).parameters:'
# The buggy guard: installs the shim on transformers 4.x too.
FLAG_GUARD = 'if not getattr(create_causal_mask, "_v5_shim", False):'

# Mask-function signatures of real transformers releases, copied from PyPI
# wheels. ``True`` marks a parameter that has a default value.
BANDS = {
    # 4.57.0 through 5.1.0 share the v4 names.
    "4.57.0/5.0.0/5.1.0 (v4 names)": {
        "pos": [("config", False), ("input_embeds", False), ("attention_mask", False),
                ("cache_position", False), ("past_key_values", False),
                ("position_ids", True), ("or_mask_function", True), ("and_mask_function", True)],
        "kwonly": [],
    },
    # 5.5.0 renamed the arg but kept cache_position (now optional) as positional.
    "5.5.0 (hybrid)": {
        "pos": [("config", False), ("inputs_embeds", False), ("attention_mask", False),
                ("cache_position", True)],
        "kwonly": [("past_key_values", True), ("position_ids", True),
                   ("or_mask_function", True), ("and_mask_function", True)],
    },
    # 5.10.0 onwards dropped cache_position.
    "5.10.1/5.15.0/5.19.0 (v5 names)": {
        "pos": [("config", False), ("inputs_embeds", False), ("attention_mask", False),
                ("past_key_values", False), ("position_ids", True), ("or_mask_function", True),
                ("and_mask_function", True), ("block_sequence_ids", True), ("layer_idx", True),
                ("allow_is_causal_skip", True)],
        "kwonly": [],
    },
}


class _FakeTensor:
    """Stands in for the tensor in cache_position; only unsqueeze is used."""

    def unsqueeze(self, dim):
        return self


def shim_block(path):
    """Extract the create_causal_mask shim (the compat block importing inspect)."""
    src = path.read_text(encoding="utf-8")
    start = 0
    while True:
        start = src.index("# --- transformers >=5 compat", start)
        end = src.index("# --- end shim", start)
        block = src[start:end + len("# --- end shim ---------------------------------------------------------------")]
        if "import inspect as _inspect" in block:
            return src, block
        start = end


def vendored_mask_kwargs(path):
    """Keyword names the vendored forward passes build for the mask calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "mask_kwargs" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys]
    raise AssertionError(f"mask_kwargs not found in {path.name}")


def stub_source(name, band):
    parts = [f"{p}=None" if has_default else p for p, has_default in band["pos"]]
    kwonly = [f"{p}=None" if has_default else p for p, has_default in band["kwonly"]]
    if kwonly:
        parts.append("*, " + ", ".join(kwonly))
    return f"def {name}({', '.join(parts)}):\n    return dict(locals())\n"


def install_shim(block, band):
    """Run the shim block with create_causal_mask stubbed to a real band signature."""
    ns = {}
    exec(stub_source("create_causal_mask", band), ns)
    exec(stub_source("create_sliding_window_causal_mask", band), ns)
    exec(block, ns)
    return ns


def call_mask_functions(ns, kwargs):
    """Make both vendored calls; return (received, type_errors)."""
    received, errors = {}, []
    try:
        received["causal"] = ns["create_causal_mask"](**kwargs)
    except TypeError as e:
        errors.append(str(e))
    try:
        received["sliding"] = ns["create_sliding_window_causal_mask"](**kwargs)
    except TypeError as e:
        errors.append(str(e))
    return received, errors


class ShimGuardTests(unittest.TestCase):
    def test_shim_probes_the_real_signature(self):
        for path in MODELING_FILES:
            with self.subTest(file=path.name):
                src, _ = shim_block(path)
                self.assertIn(PROBE_GUARD, src)
                self.assertNotIn(FLAG_GUARD, src)


class ShimBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.kwargs_by_file = {path: vendored_mask_kwargs(path) for path in MODELING_FILES}
        self.models_path, self.tokenizer_path = MODELING_FILES

    def sentinel_kwargs(self, path):
        kwargs = {name: name for name in self.kwargs_by_file[path]}
        self.assertIn("input_embeds", kwargs)
        kwargs["config"] = object()
        kwargs["cache_position"] = _FakeTensor()
        return kwargs

    def test_shim_stays_off_on_v4_signatures(self):
        # 4.57.x and 5.0/5.1 use the v4 names the vendored calls already match;
        # installing the shim there is exactly the reported TypeError.
        band = BANDS["4.57.0/5.0.0/5.1.0 (v4 names)"]
        for path in MODELING_FILES:
            with self.subTest(file=path.name):
                _, block = shim_block(path)
                ns = install_shim(block, band)
                self.assertFalse(hasattr(ns["create_causal_mask"], "_v5_shim"))
                kwargs = self.sentinel_kwargs(path)
                received, errors = call_mask_functions(ns, kwargs)
                self.assertFalse(errors)
                self.assertEqual(received["causal"]["input_embeds"], "input_embeds")
                self.assertIs(received["causal"]["cache_position"], kwargs["cache_position"])

    def test_shim_engages_and_adapts_on_v5_signatures(self):
        for band_name in ("5.5.0 (hybrid)", "5.10.1/5.15.0/5.19.0 (v5 names)"):
            with self.subTest(band=band_name):
                for path in MODELING_FILES:
                    _, block = shim_block(path)
                    ns = install_shim(block, BANDS[band_name])
                    self.assertTrue(hasattr(ns["create_causal_mask"], "_v5_shim"))
                    received, errors = call_mask_functions(ns, self.sentinel_kwargs(path))
                    self.assertFalse(errors)
                    for site in ("causal", "sliding"):
                        # input_embeds must land at the renamed parameter.
                        self.assertEqual(received[site]["inputs_embeds"], "input_embeds")
                        self.assertNotIn("input_embeds", received[site])

    def test_cache_position_is_converted_or_dropped_on_v5(self):
        for band_name in ("5.5.0 (hybrid)", "5.10.1/5.15.0/5.19.0 (v5 names)"):
            with self.subTest(band=band_name):
                # models site has no position_ids in its kwargs, so cache_position
                # becomes one.
                _, block = shim_block(self.models_path)
                ns = install_shim(block, BANDS[band_name])
                models_kwargs = self.sentinel_kwargs(self.models_path)
                received, errors = call_mask_functions(ns, models_kwargs)
                self.assertFalse(errors)
                self.assertIs(received["causal"]["position_ids"], models_kwargs["cache_position"])
                self.assertIsNot(received["causal"].get("cache_position"),
                                 models_kwargs["cache_position"])

                # tokenizer site passes position_ids itself; cache_position is
                # discarded instead of colliding with it.
                _, block = shim_block(self.tokenizer_path)
                ns = install_shim(block, BANDS[band_name])
                tokenizer_kwargs = self.sentinel_kwargs(self.tokenizer_path)
                self.assertIn("position_ids", tokenizer_kwargs)
                received, errors = call_mask_functions(ns, tokenizer_kwargs)
                self.assertFalse(errors)
                self.assertEqual(received["causal"]["position_ids"], "position_ids")
                self.assertIsNot(received["causal"].get("cache_position"),
                                 tokenizer_kwargs["cache_position"])

    def test_adapted_kwargs_cover_every_required_argument(self):
        for band_name, band in BANDS.items():
            with self.subTest(band=band_name):
                for path in MODELING_FILES:
                    required = {p for p, has_default in band["pos"] if not has_default}
                    required |= {p for p, has_default in band["kwonly"] if not has_default}
                    supplied = set(self.kwargs_by_file[path])
                    if "input_embeds" in supplied:
                        supplied.add("inputs_embeds")  # the shim renames it
                    missing = required - supplied
                    self.assertFalse(missing, f"{path.name} misses {missing} for {band_name}")


if __name__ == "__main__":
    unittest.main()
