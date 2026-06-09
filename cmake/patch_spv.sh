#!/bin/bash
set -e
for SPV in "$@"; do
  TMP=$(mktemp -d)
  ASM="$TMP/a.spvasm"
  spirv-dis "$SPV" -o "$ASM"
  grep -q 'OpLoad.*%gl_BaseInstance' "$ASM" || { rm -rf "$TMP"; continue; }
  IDX=$(grep 'OpLoad.*%gl_InstanceIndex' "$ASM" | head -1 | sed 's/^[[:space:]]*%//; s/ =.*//')
  BASE=$(grep 'OpLoad.*%gl_BaseInstance' "$ASM" | head -1 | sed 's/^[[:space:]]*%//; s/ =.*//')
  SUB=$(grep "OpISub" "$ASM" | head -1 | sed 's/^[[:space:]]*%//; s/ =.*//')
  sed -i "/OpLoad.*%gl_BaseInstance/d" "$ASM"
  sed -i "/OpISub.*$BASE/d" "$ASM"
  sed -i "s/%${SUB}/%${IDX}/g" "$ASM"
  spirv-as "$ASM" -o "$SPV"
  echo "  patched $SPV"
  rm -rf "$TMP"
done
