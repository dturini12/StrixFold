# The `dsv41` kernel kit: pack it, make its generated files, check it

The four-node lane loads a kernel kit (`TF_DS_KIT`): the family's Triton kernels as binaries (`aot/`), the EXL3
kernels (`cubins/`), `engram.json`, the RoPE tables with `rope.json`, and `vision/`. The recipe
([bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10),
folder `kit/`) holds the released kit; these tools made it and remake its generated part.

- `pack_kit.py SRC_KIT OUT_KIT [--same-as SUMS] [--title TEXT]`: copies a lane's kit and writes `MANIFEST` (sha256,
  bytes, origin, the two-node flag, path) and `SHA256SUMS`. The RoPE tables and the image bias are marked as made, not
  committed; any other file over 50 MB stops it.
- `make_rope.py MODEL KIT_DIR [--check]`: the four RoPE tables on the CPU (PyTorch), checked against `rope.json`.
- `make_bias_vl.py ORIGINAL_DIR OUT`: `vision/gate_bias_vl.safetensors`, the 43 `ffn.gate.bias_vl` tensors copied out of
  DeepSeek's original checkpoint (standard library only).
- `verify_hf_cache.py MODEL_DIR REVISION [FILE...]`: a huggingface_hub download's files (default: the index and its
  shards) against the sha256 the Hub reported at download time (`.cache/huggingface/download/*.metadata`).
- `make_vision_sidecar.py MODEL_DIR OUT_DIR [--expect IDENTITY]`: the 266 vision tower tensors (`vision.*`,
  `aligner.*`, `image_*`) byte for byte into a one-shard checkpoint (`vision.safetensors` + its index + `SHA256SUMS`),
  the directory the HIP tower opens (`TF_DS_VISION_DIR`). The 1017 tower's identity is
  `32e3e2cc4cae5628c050ccd8c8993df5531e979713a338858a437375a369fc87`, from the original or the LibertAI EXL3 checkpoint.
- `verify_kit.sh [--kit DIR] [--remake] MODEL [ORIGINAL_DIR]`: makes what is missing with the two scripts above, then
  checks every file against `MANIFEST`.
