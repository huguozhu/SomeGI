#!/usr/bin/env python3
# cmake/compile_dxil.py — 将 SPIR-V 批量转换为 DXIL
# 用法: python compile_dxil.py <spirv_dir> <dxil_output_dir> [spirv_cross_path] [dxc_path]

import subprocess, json, os, sys, glob, shutil

SPIRV_CROSS = "spirv-cross"
DXC = "dxc"

STAGE_MAP = {
    "vert": ("vs_6_5", "main"),
    "frag": ("ps_6_5", "main"),
    "comp": ("cs_6_5", "main"),
    "mesh": ("ms_6_5", "main"),
    "task": ("as_6_5", "main"),
    "rgen": ("lib_6_5", "main"),   # Ray tracing: library
    "rmiss": ("lib_6_5", "main"),
    "rchit": ("lib_6_5", "main"),
    "rahit": ("lib_6_5", "main"),
}

def detect_stage(spv_path):
    """用 spirv-cross --reflect 检测 shader stage（取首个 entry point）"""
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_spvc_reflect.json")
    result = subprocess.run(
        [SPIRV_CROSS, "--reflect", "--output", tmp, spv_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(tmp):
        return None

    try:
        with open(tmp, "r") as f:
            data = json.load(f)
        eps = data.get("entryPoints", [])
        if eps:
            mode = eps[0].get("mode", None)
            if mode and mode != "???":
                return mode
    except Exception:
        pass
    return None

def compile_one(spv_path, output_dir):
    name = os.path.splitext(os.path.basename(spv_path))[0]

    # 检测 stage
    stage = detect_stage(spv_path)
    if not stage:
        # 回退：从路径名推断
        if "frag" in name or "frag" in spv_path.lower():
            stage = "frag"
        elif "vert" in name or "vert" in spv_path.lower():
            stage = "vert"
        elif "comp" in name or "comp" in spv_path.lower():
            stage = "comp"
        elif "mesh" in name or "mesh" in spv_path.lower():
            stage = "mesh"
        elif "task" in name or "task" in spv_path.lower():
            stage = "task"
        else:
            stage = "comp"  # 默认 compute

    profile, entry = STAGE_MAP.get(stage, ("cs_6_5", "main"))

    # 保持相同的子目录结构
    rel_dir = os.path.relpath(os.path.dirname(spv_path), spirv_dir)
    out_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    hlsl_path = os.path.join(out_dir, f"{name}.hlsl")
    dxil_path = os.path.join(out_dir, f"{name}.dxil")

    # Step 1: SPIR-V → HLSL
    ret = subprocess.run([
        SPIRV_CROSS, "--hlsl", "--shader-model", "65",
        "--output", hlsl_path, spv_path
    ], capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"  FAIL spirv-cross: {name} — {ret.stderr.strip()[:200]}")
        return False

    # Step 2: HLSL → DXIL
    ret = subprocess.run([
        DXC, "-T", profile, "-E", entry,
        "-Fo", dxil_path, hlsl_path
    ], capture_output=True, text=True)

    if ret.returncode != 0:
        err = ret.stderr.strip()
        # Mesh shader SPIRV-Cross 转换暂不支持，允许失败
        if "mesh" in name or "task" in name:
            print(f"  SKIP {name} (mesh/task shader not yet supported by SPIRV-Cross)")
            return True  # 不计入失败
        print(f"  FAIL dxc: {name} (profile={profile}) — {err[:200]}")
        return False

    # 清理中间 .hlsl 文件
    try: os.remove(hlsl_path)
    except: pass

    print(f"  OK {name}.dxil ({os.path.getsize(dxil_path)} bytes)")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compile_dxil.py <spirv_dir> <dxil_output_dir> [spirv_cross] [dxc]")
        sys.exit(1)

    spirv_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        SPIRV_CROSS = sys.argv[3]
    if len(sys.argv) > 4:
        DXC = sys.argv[4]

    spv_files = glob.glob(os.path.join(spirv_dir, "**/*.spv"), recursive=True)
    if not spv_files:
        print(f"No .spv files found in {spirv_dir}")
        sys.exit(1)

    ok = 0
    fail = 0
    for f in sorted(spv_files):
        if compile_one(f, output_dir):
            ok += 1
        else:
            fail += 1

    print(f"\nDXIL compilation: {ok} OK, {fail} FAILED (of {len(spv_files)})")
    # 编译失败仅为 warning（mesh/task shader 等暂不支持），不影响构建
