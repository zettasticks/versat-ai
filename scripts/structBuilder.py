from versatDefs import *

import struct

# TODO: Maybe just move this to versatCommon.py


def EndianessToStructArg(endianess: Endianess):
    endianArg = ""
    if endianess == Endianess.NATIVE:
        endianArg = "@"
    elif endianess == Endianess.LITTLE_ENDIAN:
        endianArg = "<"
    elif endianess == Endianess.BIG_ENDIAN:
        endianArg = ">"
    else:
        assert False

    return endianArg


def NormalizeArray(arr, minElementCount=-1):
    if minElementCount == -1:
        minElementCount = len(arr)

    if len(arr) < minElementCount:
        arr += [0] * (minElementCount - len(arr))

    return arr


class StructBuilder:
    __slots__ = [
        "content",
        "u32Packer",
        "u64Packer",
        "i32Packer",
        "i64Packer",
        "f32Packer",
    ]

    def __init__(self, endianess: Endianess = Endianess.NATIVE):
        endianArg = EndianessToStructArg(endianess)

        self.content = bytearray()
        self.i32Packer = struct.Struct(f"{endianArg}i")
        self.u32Packer = struct.Struct(f"{endianArg}I")
        self.i64Packer = struct.Struct(f"{endianArg}q")
        self.u64Packer = struct.Struct(f"{endianArg}Q")
        self.f32Packer = struct.Struct(f"{endianArg}f")

    def U32(self, val):
        self.content += self.u32Packer.pack(val)

    def I32(self, val):
        self.content += self.i32Packer.pack(val)

    def U64(self, val):
        self.content += self.u64Packer.pack(val)

    def I64(self, val):
        self.content += self.i64Packer.pack(val)

    def F32(self, val):
        self.content += self.f32Packer.pack(val)

    def PrependU32(self, val):
        self.content = self.u32Packer.pack(val) + self.content

    def I32Array(self, arr, minElementCount=-1):
        arr = NormalizeArray(arr, minElementCount)
        for x in arr:
            self.I32(x)

    def I64Array(self, arr, minElementCount=-1):
        arr = NormalizeArray(arr, minElementCount)
        for x in arr:
            self.I64(x)

    def DataSource(self, sourceType, val):
        self.U32(sourceType)
        self.U32(val)

    def Append(self, val):
        self.content += val

    def GetSize(self):
        return len(self.content)

    def GetContent(self):
        return bytearray(self.content)
