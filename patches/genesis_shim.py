import sys, os, shutil

# Find the installed vllm package location
vllm_site = None
for p in sys.path:
    candidate = os.path.join(p, "vllm")
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "__init__.py")):
        vllm_site = candidate
        break

if not vllm_site:
    raise RuntimeError("Could not find installed vllm package")

# Copy _genesis into it
src = "/patches/genesis/vllm/_genesis"
dst = os.path.join(vllm_site, "_genesis")
if not os.path.exists(dst):
    shutil.copytree(src, dst)
    print(f"[genesis_shim] Copied _genesis into {vllm_site}")
else:
    print(f"[genesis_shim] _genesis already present at {dst}")

from vllm._genesis.patches.apply_all import main
main()
