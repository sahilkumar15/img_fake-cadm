#!/usr/bin/env python3
"""
lib/fix_ldm_paths.py
─────────────────────
Universally fixes ldm.json key paths for ANY dataset (FF++, CelebDF, DFD,
DiffSwap, WildDeepfake, etc.) by doing a simple prefix replacement.

How it works:
  1. Reads the first key from ldm.json to find the current prefix
  2. The prefix = everything before the first subfolder that exists on disk
  3. Replaces the old prefix with the desired img_path in all keys + source_paths

USAGE:
  python lib/fix_ldm_paths.py \
      --ldm      ./test_images-2/test_images_celebdf/ldm.json \
      --img_path ./test_images-2/test_images_celebdf

  # Fix all datasets at once:
  python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images/ldm.json              --img_path ./test_images-2/test_images
  python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_celebdf/ldm.json      --img_path ./test_images-2/test_images_celebdf
  python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_dfd/ldm.json          --img_path ./test_images-2/test_images_dfd
  python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_diffswap/ldm.json     --img_path ./test_images-2/test_images_diffswap
  python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_wilddeepfake/ldm.json --img_path ./test_images-2/test_images_wilddeepfake
"""

import json
import os
import argparse


def find_old_prefix(sample_key, img_path):
    """
    Find the old prefix in sample_key by splitting on '/' and finding
    where the path diverges from what's actually inside img_path.

    Strategy: the old prefix is everything up to (not including) the first
    path component that is also a subdirectory of img_path.
    """
    img_path = img_path.rstrip("/")

    # Get the top-level subdirectories that actually exist in img_path
    try:
        subdirs = {d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))}
    except Exception as e:
        print(f"ERROR: Cannot list {img_path}: {e}")
        return None

    if not subdirs:
        print(f"ERROR: No subdirectories found in {img_path}")
        return None

    print(f"  Subdirs in img_path: {sorted(subdirs)}")

    # Split the sample key into parts and find where a known subdir appears
    parts = sample_key.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in subdirs:
            # Everything before index i is the old prefix
            old_prefix = "/".join(parts[:i]).rstrip("/")
            return old_prefix

    # Fallback: try to find any part of the key that matches a subdir name
    # even if nested (e.g. DFD_ prefix case)
    for i, part in enumerate(parts):
        for subdir in subdirs:
            if part.startswith(subdir) or subdir.startswith(part):
                old_prefix = "/".join(parts[:i]).rstrip("/")
                return old_prefix

    return None


def fix_ldm(ldm_path, img_path):
    img_path = img_path.rstrip("/")
    # Normalise to avoid ./path vs path mismatches
    img_path_norm = os.path.normpath(img_path)

    print(f"\nLoading: {ldm_path}")
    with open(ldm_path) as f:
        ldm = json.load(f)

    if not ldm:
        print("ldm.json is empty — nothing to fix.")
        return

    sample_key = next(iter(ldm))
    print(f"Sample key : {sample_key}")
    print(f"img_path   : {img_path}")

    # Normalise sample key for comparison
    sample_norm = os.path.normpath(sample_key)
    img_norm    = os.path.normpath(img_path)

    # Check if already correct by seeing if the key's path starts with img_path
    # after normalisation
    if sample_norm.startswith(img_norm + os.sep) or sample_norm.startswith(img_norm + "/"):
        print("✓ Paths already match — no fix needed.")
        return

    # Find old prefix
    old_prefix = find_old_prefix(sample_key, img_path)

    if old_prefix is None:
        # Last resort: take everything before the second-to-last component
        # that doesn't exist as a real path
        parts = sample_key.replace("\\", "/").split("/")
        # Try progressively shorter prefixes
        for cut in range(len(parts) - 1, 0, -1):
            candidate_prefix = "/".join(parts[:cut])
            remainder = "/".join(parts[cut:])
            test_path = img_path + "/" + remainder + ".png"
            if os.path.exists(test_path):
                old_prefix = candidate_prefix
                print(f"  Found prefix via file check: {old_prefix!r}")
                break

    if old_prefix is None:
        print("ERROR: Could not determine old prefix automatically.")
        print("Manual fix — open ldm.json and check what prefix the keys start with.")
        print(f"Then run:")
        print(f"  python lib/fix_ldm_paths.py --ldm {ldm_path} --img_path {img_path} --old_prefix <PREFIX>")
        return

    desired_prefix = img_path
    print(f"Old prefix : {old_prefix!r}")
    print(f"New prefix : {desired_prefix!r}")

    if old_prefix == desired_prefix:
        print("✓ Prefixes already match — no fix needed.")
        return

    # Rewrite all keys and source_paths
    print(f"Rewriting {len(ldm)} entries ...")
    new_ldm = {}
    fixed_sources = 0

    for key, val in ldm.items():
        # Fix key
        if key.startswith(old_prefix):
            new_key = desired_prefix + key[len(old_prefix):]
        else:
            # Try normalised comparison
            key_norm = os.path.normpath(key)
            old_norm = os.path.normpath(old_prefix)
            if key_norm.startswith(old_norm):
                remainder = key_norm[len(old_norm):]
                new_key = desired_prefix + remainder
            else:
                new_key = key  # leave unchanged if can't match

        # Fix source_path
        sp = val.get("source_path", "")
        if sp:
            if sp.startswith(old_prefix):
                val = dict(val)
                val["source_path"] = desired_prefix + sp[len(old_prefix):]
                fixed_sources += 1
            else:
                sp_norm = os.path.normpath(sp)
                old_norm = os.path.normpath(old_prefix)
                if sp_norm.startswith(old_norm):
                    val = dict(val)
                    val["source_path"] = desired_prefix + sp_norm[len(old_norm):]
                    fixed_sources += 1

        new_ldm[new_key] = val

    # Verify a sample exists on disk
    sample_new_key = next(iter(new_ldm))
    sample_png = sample_new_key + ".png"
    if os.path.exists(sample_png):
        print(f"✓ Verified on disk: {sample_png}")
    else:
        print(f"⚠ Not found on disk: {sample_png}")
        print("  (This may be OK if that specific frame had no face detected)")

    # Backup and write
    backup = ldm_path + ".bak"
    if not os.path.exists(backup):  # don't overwrite existing backup
        os.rename(ldm_path, backup)
        print(f"  Backup: {backup}")
    else:
        os.replace(ldm_path, backup)
        print(f"  Backup updated: {backup}")

    with open(ldm_path, "w") as f:
        json.dump(new_ldm, f)

    print(f"✓ Fixed {len(new_ldm)} keys, {fixed_sources} source_paths → {ldm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm",        required=True,
                        help="Path to ldm.json")
    parser.add_argument("--img_path",   required=True,
                        help="The img_path value from your test config")
    parser.add_argument("--old_prefix", default=None,
                        help="Override old prefix (use if auto-detection fails)")
    args = parser.parse_args()

    if not os.path.exists(args.ldm):
        print(f"ERROR: {args.ldm} not found")
        return
    if not os.path.isdir(args.img_path):
        print(f"ERROR: {args.img_path} is not a directory")
        return

    if args.old_prefix:
        # Manual override mode
        img_path = args.img_path.rstrip("/")
        print(f"Loading: {args.ldm}")
        with open(args.ldm) as f:
            ldm = json.load(f)
        sample = next(iter(ldm))
        print(f"Sample key : {sample}")
        print(f"Old prefix : {args.old_prefix!r}")
        print(f"New prefix : {img_path!r}")
        new_ldm = {}
        for k, v in ldm.items():
            nk = img_path + k[len(args.old_prefix):] if k.startswith(args.old_prefix) else k
            sp = v.get("source_path", "")
            if sp and sp.startswith(args.old_prefix):
                v = dict(v)
                v["source_path"] = img_path + sp[len(args.old_prefix):]
            new_ldm[nk] = v
        backup = args.ldm + ".bak"
        os.replace(args.ldm, backup)
        with open(args.ldm, "w") as f:
            json.dump(new_ldm, f)
        print(f"✓ Fixed {len(new_ldm)} keys → {args.ldm}")
        return

    fix_ldm(args.ldm, args.img_path)


if __name__ == "__main__":
    main()