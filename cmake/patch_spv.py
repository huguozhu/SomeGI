#!/usr/bin/env python3
"""Patch SPIR-V: remove gl_BaseInstance subtraction from SV_InstanceID."""

import subprocess, sys, os, tempfile, re

def find_tool(name):
    sdk = os.environ.get("VULKAN_SDK", "")
    if sdk:
        for ext in ("", ".exe"):
            p = os.path.join(sdk, "Bin", name + ext)
            if os.path.exists(p): return p
    return name

def patch_one(spv):
    d = find_tool("spirv-dis")
    a = find_tool("spirv-as")
    td = tempfile.mkdtemp()
    asm = os.path.join(td, "a.spvasm")
    r = subprocess.run([d, spv, "-o", asm], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  dis failed", file=sys.stderr); return False

    with open(asm) as f:
        lines = f.readlines()

    # Phase 1: find IDs
    idx_id = base_id = base_var_id = sub_id = None
    for line in lines:
        m = re.match(r'\s*%(\w+)\s*=\s*OpLoad\s+%int\s+%(\w+)', line)
        if m:
            if 'gl_BaseInstance' in line: base_id, base_var_id = m.group(1), m.group(2)
            elif 'gl_InstanceIndex' in line: idx_id = m.group(1)

    if base_id is None:
        return True  # already patched

    # Phase 2: filter lines + find ISub
    out = []
    for line in lines:
        # Skip BaseInstance OpLoad
        if re.match(rf'\s*%{base_id}\s*=\s*OpLoad\s+%int\s+%gl_BaseInstance', line):
            continue
        # Skip OpDecorate %gl_BaseInstance
        if re.match(r'\s*OpDecorate\s+%gl_BaseInstance\s+BuiltIn\s+BaseInstance', line):
            continue
        # Skip OpVariable %gl_BaseInstance
        if re.match(r'\s*%gl_BaseInstance\s*=\s*OpVariable\s+', line):
            continue
        # Fix OpEntryPoint: remove %gl_BaseInstance reference
        if 'OpEntryPoint' in line and '%gl_BaseInstance' in line:
            line = re.sub(r'\s*%gl_BaseInstance\b', '', line)

        # Find ISub
        m = re.match(rf'\s*%(\w+)\s*=\s*OpISub\s+%int\s+%{idx_id}\s+%{base_id}', line)
        if m:
            sub_id = m.group(1)
            continue  # skip ISub

        out.append(line)

    if sub_id is None:
        print(f"  ISub not found", file=sys.stderr); return False

    # Phase 3: replace %sub_id with %idx_id
    for i, line in enumerate(out):
        out[i] = re.sub(rf'%{sub_id}(?![0-9])', f'%{idx_id}', line)

    with open(asm, 'w') as f:
        f.writelines(out)

    r = subprocess.run([a, asm, "-o", spv], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  as failed: {r.stderr.strip()[:200]}", file=sys.stderr); return False

    print(f"  patched {os.path.basename(spv)}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_spv.py <file.spv>..."); sys.exit(1)
    ok = True
    for spv in sys.argv[1:]:
        if not os.path.exists(spv): continue
        if not patch_one(spv): ok = False
    if not ok: print("WARNING: some patches failed", file=sys.stderr)
