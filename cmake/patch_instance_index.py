#!/usr/bin/env python3
"""
Post-process a SPIR-V binary: remove the gl_BaseInstance subtraction
so that SV_InstanceID maps to gl_InstanceIndex directly (not instanceID).

Slang generates:
    %base = OpLoad %int %gl_BaseInstance
    %idx  = OpLoad %int %gl_InstanceIndex
    %sub  = OpISub %int %idx %base
    %u    = OpBitcast %uint %sub

We replace the OpISub result with the InstanceIndex load result,
so the shader uses gl_InstanceIndex (= firstInstance + instanceID) directly.

Usage: python patch_instance_index.py <input.spv> <output.spv>
"""

import struct
import sys
import os

OP_LOAD        = 61
OP_ISUB        = 80
OP_BITCAST     = 124

def patch_spirv(path: str, outpath: str):
    with open(path, 'rb') as f:
        data = f.read()
    words = list(struct.unpack(f'<{len(data)//4}I', data))

    # Find %gl_InstanceIndex and %gl_BaseInstance IDs
    instance_id = None
    base_id = None
    i = 0
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if op == 59:  # OpVariable
            type_id = words[i + 1]
            result_id = words[i + 2]
            sc = words[i + 3]
            # Look for BuiltIn InstanceIndex / BaseInstance decorations
            # We just check by name (slang always names them consistently)
            pass
        i += 1

    # Simpler approach: scan all decorations for BuiltIn
    for i, w in enumerate(words):
        op = w & 0xFFFF
        wc = w >> 16
        if op == 71 and wc >= 4:  # OpDecorate
            if words[i + 2] == 11:  # BuiltIn
                builtin = words[i + 3]
                if builtin == 2:  # InstanceIndex
                    instance_id = words[i + 1]
                elif builtin == 1:  # BaseInstance
                    base_id = words[i + 1]

    if instance_id is None or base_id is None:
        print("patch_instance_index: InstanceIndex or BaseInstance not found, skipping", file=sys.stderr)
        return

    # Now find the OpISub that uses both
    sub_result = None
    idx_load_result = None
    i = 0
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if op == OP_ISUB and wc >= 5:
            rtype = words[i + 1]
            result = words[i + 2]
            op1 = words[i + 3]
            op2 = words[i + 4]
            # Check if op1 or op2 is a load from instance_id/base_id
            # We need the load result IDs
            pass
        i += 1

    # Simpler: scan all OpLoad instructions to find the loads of instance/base
    # Then find the OpISub that consumes them
    load_base = None
    load_idx = None
    i = 0
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if op == OP_LOAD and wc >= 4:
            result = words[i + 2]
            ptr = words[i + 3]
            if ptr == base_id:
                load_base = result
            elif ptr == instance_id:
                load_idx = result
        i += 1

    if load_base is None or load_idx is None:
        print("patch_instance_index: load instructions not found, skipping", file=sys.stderr)
        return

    # Find OpISub that uses load_base
    i = 0
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if op == OP_ISUB and wc >= 5:
            op1 = words[i + 3]
            op2 = words[i + 4]
            if op2 == load_base and op1 == load_idx:
                sub_result = words[i + 2]
                break
        i += 1

    if sub_result is None:
        print("patch_instance_index: OpISub not found, skipping", file=sys.stderr)
        return

    # Now replace all uses of sub_result with load_idx.
    # Also convert from int to uint where needed (Bitcast).
    # Strategy: replace sub_result ID with load_idx in all instructions.
    # Then fix the OpBitcast to cast to uint.
    out = []
    i = 0
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if op == OP_ISUB and wc >= 5 and words[i + 2] == sub_result:
            # Remove the ISub instruction entirely
            # Replace with OpCopyObject: %sub_result = OpCopyObject %rtype %load_idx
            out.append((4 << 16) | 83)  # OpCopyObject, 4 words
            out.append(words[i + 1])     # result type
            out.append(sub_result)       # result id (keep same)
            out.append(load_idx)         # operand
            i += wc
            continue
        if op == OP_BITCAST and wc >= 5:
            if words[i + 3] == sub_result:
                # Change: %u = OpBitcast %uint %sub_result
                # To:     %u = OpBitcast %uint %load_idx
                out.append(words[i])
                out.append(words[i + 1])
                out.append(words[i + 2])
                out.append(load_idx)  # use load_idx instead of sub_result
                i += wc
                continue
        if op == OP_LOAD and wc >= 4 and words[i + 3] == base_id:
            # Remove the BaseInstance load (unused after our patch)
            # Replace with OpNop or just skip
            i += wc
            continue
        # For all other instructions, replace sub_result references with load_idx
        new_wc = wc
        new_words = [words[i]]
        for j in range(1, wc):
            val = words[i + j]
            if val == sub_result:
                val = load_idx
            new_words.append(val)
        out.extend(new_words)
        i += wc

    result = struct.pack(f'<{len(out)}I', *out)
    with open(outpath, 'wb') as f:
        f.write(result)
    print(f"patched {path} -> {outpath} (removed BaseInstance subtraction)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_instance_index.py <input.spv> [output.spv]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    patch_spirv(src, dst)
