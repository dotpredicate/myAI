import struct
import tempfile
import unittest
from pathlib import Path

from backend.inference.llama_cpp.gguf import find_cached_gguf, list_cached_quants, resolve_cached_alias
from backend.inference.llama_cpp.gguf_header import (
    GgufParseError,
    TruncatedGgufError,
    parse_gguf_header,
    read_gguf_header,
)


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    payload = _string(key) + struct.pack("<I", 9)
    payload += struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for value in values:
        payload += _string(value)
    return payload


def _build_gguf() -> tuple[bytes, int]:
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", 1)  # tensor count
    body += struct.pack("<Q", 4)  # metadata count
    body += _kv_string("general.architecture", "llama")
    body += _kv_uint32("llama.context_length", 8192)
    body += _kv_uint32("general.alignment", 32)
    body += _kv_string_array("tokenizer.ggml.tokens", ["a", "bb"])
    body += _string("blk.0.weight")
    body += struct.pack("<I", 2)  # dimensions
    body += struct.pack("<Q", 2) + struct.pack("<Q", 3)
    body += struct.pack("<I", 0)  # tensor type
    body += struct.pack("<Q", 0)  # tensor offset
    tensor_data_offset = (len(body) + 31) // 32 * 32
    return body, tensor_data_offset


class TestGgufHeaderParser(unittest.TestCase):
    def test_parses_metadata_tensors_and_alignment(self) -> None:
        body, tensor_data_offset = _build_gguf()
        data = body + b"\0" * (tensor_data_offset - len(body)) + b"\0" * 64

        header = parse_gguf_header(data)

        self.assertEqual(header.size, len(data))
        self.assertEqual(header.architecture, "llama")
        self.assertEqual(header.context_length, 8192)
        self.assertEqual(header.parameters, 6)
        self.assertEqual(header.tensor_data_offset, tensor_data_offset)

    def test_reads_header_from_file(self) -> None:
        body, tensor_data_offset = _build_gguf()
        data = body + b"\0" * (tensor_data_offset - len(body)) + b"\0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(data)

            header = read_gguf_header(path)

        self.assertEqual(header.size, len(data))
        self.assertEqual(header.tensor_data_offset, tensor_data_offset)

    def test_rejects_bad_magic(self) -> None:
        with self.assertRaises(GgufParseError):
            parse_gguf_header(b"NOPE" + b"\0" * 64)

    def test_rejects_unsupported_version(self) -> None:
        with self.assertRaises(GgufParseError):
            parse_gguf_header(b"GGUF" + struct.pack("<I", 1) + b"\0" * 64)

    def test_reports_truncated_header(self) -> None:
        body, _ = _build_gguf()
        with self.assertRaises(TruncatedGgufError):
            parse_gguf_header(body[:16])


class TestLocalGgufCache(unittest.TestCase):
    def _make_cache(self, root: Path) -> Path:
        snapshot = root / "models--org--model" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (root / "models--org--model" / "refs").mkdir()
        (root / "models--org--model" / "refs" / "main").write_text("abc123")
        (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"x")
        (snapshot / "Model-Q8_0.gguf").write_bytes(b"x")
        (snapshot / "mmproj-F16.gguf").write_bytes(b"x")
        return snapshot

    def test_finds_cached_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._make_cache(root)

            found = find_cached_gguf("org/model", "Model-Q4_K_M.gguf", root)

        self.assertEqual(found, snapshot / "Model-Q4_K_M.gguf")

    def test_lists_only_real_quants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_cache(root)

            quants = list_cached_quants("org/model", root)

        self.assertEqual(quants, [("Model-Q4_K_M.gguf", "Q4_K_M"), ("Model-Q8_0.gguf", "Q8_0")])

    def test_resolves_cached_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_cache(root)

            resolved = resolve_cached_alias("org/model:Q8_0", root)

        self.assertEqual(resolved, ("org/model", "Model-Q8_0.gguf"))
