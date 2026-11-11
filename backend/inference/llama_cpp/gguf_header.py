import io
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
MAX_ARRAY_ITEMS = 10_000_000

# GGUF metadata value types.
_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_FIXED_SIZES = {
    _TYPE_UINT8: 1,
    _TYPE_INT8: 1,
    _TYPE_UINT16: 2,
    _TYPE_INT16: 2,
    _TYPE_UINT32: 4,
    _TYPE_INT32: 4,
    _TYPE_FLOAT32: 4,
    _TYPE_BOOL: 1,
    _TYPE_UINT64: 8,
    _TYPE_INT64: 8,
    _TYPE_FLOAT64: 8,
}

_FIXED_FORMATS: dict[int, str] = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}


class GgufParseError(ValueError):
    pass


class TruncatedGgufError(GgufParseError):
    pass


@dataclass(frozen=True)
class GgufHeader:
    size: int
    parameters: int
    architecture: str
    context_length: int
    tensor_data_offset: int


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self._stream = stream
        self._size = size

    def tell(self) -> int:
        return self._stream.tell()

    def _check(self, length: int) -> None:
        if length < 0 or self.tell() + length > self._size:
            raise TruncatedGgufError("GGUF header is truncated")

    def read(self, length: int) -> bytes:
        self._check(length)
        data = self._stream.read(length)
        if len(data) != length:
            raise TruncatedGgufError("GGUF header is truncated")
        return data

    def skip(self, length: int) -> None:
        self._check(length)
        self._stream.seek(length, io.SEEK_CUR)

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        self._check(length)
        return self.read(length).decode("utf-8", errors="replace")

    def skip_value(self, value_type: int) -> None:
        if value_type == _TYPE_STRING:
            self.skip(self.u64())
            return
        if value_type == _TYPE_ARRAY:
            element_type = self.u32()
            count = self.u64()
            if count > MAX_ARRAY_ITEMS:
                raise GgufParseError(f"GGUF array is too large ({count} items)")
            if element_type == _TYPE_STRING:
                for _ in range(count):
                    self.skip(self.u64())
            elif element_type == _TYPE_ARRAY:
                for _ in range(count):
                    self.skip_value(_TYPE_ARRAY)
            elif element_type in _FIXED_SIZES:
                self.skip(count * _FIXED_SIZES[element_type])
            else:
                raise GgufParseError(f"Unknown GGUF array element type {element_type}")
            return
        size = _FIXED_SIZES.get(value_type)
        if size is None:
            raise GgufParseError(f"Unknown GGUF value type {value_type}")
        self.skip(size)

    def read_value(self, value_type: int) -> Any:
        if value_type == _TYPE_STRING:
            return self.string()
        fmt = _FIXED_FORMATS.get(value_type)
        if fmt is None:
            return None
        return struct.unpack(fmt, self.read(_FIXED_SIZES[value_type]))[0]


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def _valid_alignment(value: Any) -> int:
    if isinstance(value, int) and value > 0 and value & (value - 1) == 0:
        return value
    return DEFAULT_ALIGNMENT


def _parse_header(reader: _Reader, file_size: int) -> GgufHeader:
    if reader.read(4) != GGUF_MAGIC:
        raise GgufParseError("Not a GGUF file")

    version = reader.u32()
    if version not in SUPPORTED_VERSIONS:
        raise GgufParseError(f"Unsupported GGUF version {version}")

    tensor_count = reader.u64()
    kv_count = reader.u64()
    if tensor_count > 10_000_000 or kv_count > 10_000_000:
        raise GgufParseError("GGUF header has an unreasonable item count")

    architecture = ""
    context_length = 0
    alignment = DEFAULT_ALIGNMENT

    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.u32()
        if key == "general.architecture":
            value = reader.read_value(value_type)
            if isinstance(value, str):
                architecture = value
        elif key == "general.alignment":
            alignment = _valid_alignment(reader.read_value(value_type))
        elif key.endswith(".context_length"):
            value = reader.read_value(value_type)
            if isinstance(value, int) and value > 0:
                context_length = value
        else:
            reader.skip_value(value_type)

    parameters = 0
    for _ in range(tensor_count):
        reader.string()  # tensor name
        dimensions = reader.u32()
        elements = 1
        for _ in range(dimensions):
            elements *= reader.u64()
        reader.u32()  # tensor type
        reader.u64()  # tensor offset
        parameters += elements

    tensor_data_offset = _align_up(reader.tell(), alignment)
    return GgufHeader(
        size=file_size,
        parameters=parameters,
        architecture=architecture,
        context_length=context_length,
        tensor_data_offset=tensor_data_offset,
    )


def parse_gguf_header(data: bytes, *, file_size: int | None = None) -> GgufHeader:
    return _parse_header(_Reader(io.BytesIO(data), len(data)), file_size or len(data))


def read_gguf_header(path: Path) -> GgufHeader:
    file_size = path.stat().st_size
    if file_size <= 0:
        raise GgufParseError("GGUF file is empty")
    with path.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return _parse_header(_Reader(cast(BinaryIO, mapped), file_size), file_size)
