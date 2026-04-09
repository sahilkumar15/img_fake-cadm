#!/usr/bin/env python3
"""
lib/fix_dfd.py  —  fixes the DFD ldm.json path issue.

The problem: previous fix run created doubled paths like:
  ./test_images-2/test_images_dfd/DFD_manipulated_sequences/DFD_manipulated_sequences/...

This script restores from .bak and applies the correct fix.

Run:
  python lib/fix_dfd.py
"""

import json, os, shutil

LDM_PATH  = "./test_images-2/test_images_dfd/ldm.json"
BAK_PATH  = "./test_images-2/test_images_dfd/ldm.json.bak"
IMG_PATH  = "./test_images-2/test_images_dfd"

# ── Step 1: figure out which file to use as source ───────────────────────────

if os.path.exists(BAK_PATH):
    print(f"Using backup: {BAK_PATH}")
    src = BAK_PATH
else:
    print(f"No backup found, using: {LDM_PATH}")
    src = LDM_PATH

with open(src) as f:
    ldm = json.load(f)

sample = next(iter(ldm))
print(f"Sample key in source: {sample}")

# ── Step 2: detect old prefix ─────────────────────────────────────────────────
# DFD keys look like one of:
#   ./test_images_dfd/DFD_original_sequences/...      (original)
#   ./test_images-2/test_images_dfd/DFD_manipulated_sequences/DFD_...  (broken double)
#
# The subdirs that actually exist on disk:
try:
    subdirs = sorted(os.listdir(IMG_PATH))
    print(f"Actual subdirs in {IMG_PATH}: {subdirs}")
except Exception as e:
    print(f"Cannot list {IMG_PATH}: {e}")
    raise SystemExit(1)

# Find what prefix the key uses before the first real subdir
parts = sample.replace("\\", "/").split("/")
old_prefix = None
for i, part in enumerate(parts):
    if any(part == s or part == s.rstrip("/") for s in subdirs):
        old_prefix = "/".join(parts[:i])
        break

if old_prefix is None:
    # Try stripping known doubled segment
    # Key: ./test_images-2/test_images_dfd/DFD_X/DFD_X/...
    # Find first occurrence of a subdir name
    for i, part in enumerate(parts):
        for s in subdirs:
            if part.startswith(s[:3]):  # rough prefix match e.g. "DFD"
                old_prefix = "/".join(parts[:i])
                break
        if old_prefix is not None:
            break

print(f"Detected old prefix: {old_prefix!r}")
print(f"Desired prefix:      {IMG_PATH!r}")

if old_prefix == IMG_PATH:
    print("✓ Already correct — no fix needed.")
    raise SystemExit(0)

# ── Step 3: rewrite all keys ──────────────────────────────────────────────────
print(f"Rewriting {len(ldm)} entries...")
new_ldm = {}
errors  = 0

for key, val in ldm.items():
    val = dict(val)

    if key.startswith(old_prefix):
        new_key = IMG_PATH + key[len(old_prefix):]
    else:
        new_key = key
        errors += 1

    sp = val.get("source_path", "")
    if sp.startswith(old_prefix):
        val["source_path"] = IMG_PATH + sp[len(old_prefix):]

    new_ldm[new_key] = val

# ── Step 4: verify a sample ───────────────────────────────────────────────────
sample_new = next(iter(new_ldm))
png = sample_new + ".png"
if os.path.exists(png):
    print(f"✓ Verified on disk: {png}")
else:
    print(f"⚠ Not on disk: {png}")
    print(f"  Checking a few more keys...")
    found = False
    for k in list(new_ldm.keys())[:20]:
        if os.path.exists(k + ".png"):
            print(f"  ✓ Found: {k}.png")
            found = True
            break
    if not found:
        print(f"  ⚠ None of first 20 keys found — check IMG_PATH is correct")

if errors:
    print(f"⚠ {errors} keys could not be matched to old prefix")

# ── Step 5: save ─────────────────────────────────────────────────────────────
# Always keep a fresh backup of the .bak
shutil.copy2(src, BAK_PATH)
with open(LDM_PATH, "w") as f:
    json.dump(new_ldm, f)

print(f"\n✓ Done! {len(new_ldm)} keys written to {LDM_PATH}")
print(f"\nNow run:")
print(f"  CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test_dfd.cfg")


"""

python -c "

import json

ldm = json.load(open('./test_images-2/test_images_dfd/ldm.json.bak'))

old = './test_images_dfd'

new = './test_images-2/test_images_dfd'

fixed = {}

for k,v in ldm.items():

    v = dict(v)

    nk = new + k[len(old):] if k.startswith(old) else k

    if v.get('source_path','').startswith(old):

        v['source_path'] = new + v['source_path'][len(old):]

    fixed[nk] = v

json.dump(fixed, open('./test_images-2/test_images_dfd/ldm.json','w'))

print(f'Done: {len(fixed)} keys fixed')

"


"""