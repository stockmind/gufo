#!/usr/bin/env python3
"""Convert ACE-Step 1.5 Hugging Face checkpoints into gufo's single-file layout.

gufo's ACE-Step files are plain safetensors with the Hugging Face tensor names
(the ComfyUI split files are exactly that): the XL DiT keeps all 829 tensors of
`model.safetensors.index.json` (decoder.*, encoder.*, tokenizer.*,
detokenizer.*, null_condition_emb) as BF16 in one file, the 5 Hz LM keeps the
Qwen3 tensors (model.embed_tokens.weight, model.layers.N.*, model.norm.weight
and, when untied, lm_head.weight). This script merges the sharded F32 / BF16
Hugging Face files into such a file, casting to BF16 (round to nearest even),
and records the variant in the safetensors metadata (`gufo.variant`), which
`gufo serve music` reads to pick the sampler. No torch needed; numpy only.

Usage:
  python tools/ace_step/convert_hf_to_gufo.py dit  SRC_DIR OUT.safetensors
  python tools/ace_step/convert_hf_to_gufo.py lm   SRC_DIR OUT.safetensors
  python tools/ace_step/convert_hf_to_gufo.py silence silence_latent.pt OUT

  dit      an `acestep-v15-xl-{turbo,sft,base}` directory (config.json,
           model-0000N-of-0000M.safetensors) -> dit-<variant>.safetensors
  lm       an `acestep-5Hz-lm-{0.6B,1.7B,4B}` directory -> lm-<size>.safetensors
  silence  the checkpoint's silence_latent.pt (needs torch) ->
           silence_latent.safetensors with the tensor `silence_latent`
           [frames, 64] in F32 (the same for every variant: it is a VAE
           property, so the existing file can be reused)

Example (model directory D:\\llm-modelle\\ACE-Step\\gufo):
  python tools/ace_step/convert_hf_to_gufo.py dit \\
      D:\\llm-modelle\\ACE-Step\\acestep-v15-xl-sft \\
      D:\\llm-modelle\\ACE-Step\\gufo\\dit-sft.safetensors
  python tools/ace_step/convert_hf_to_gufo.py lm \\
      D:\\llm-modelle\\ACE-Step\\Ace-Step1.5\\acestep-5Hz-lm-1.7B \\
      D:\\llm-modelle\\ACE-Step\\gufo\\lm-1.7b.safetensors
  gufo serve music --model D:\\llm-modelle\\ACE-Step\\gufo --dit sft --lm 1.7b

Options:
  --keep-dtype   keep F32 tensors as F32 instead of casting to BF16
  --force        overwrite an existing output file

Written with AI assistance (Claude Code).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


def read_header(path: Path) -> tuple[dict, int]:
    """Returns (header json, data start offset) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def bf16_from_f32(block: np.ndarray) -> np.ndarray:
    """F32 -> BF16 bits, round to nearest even (NaN keeps a quiet bit)."""
    bits = block.astype(np.float32).view(np.uint32)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    quiet = ((bits >> 16) | 0x40).astype(np.uint16)
    return np.where(nan, quiet, rounded)


def shard_files(src: Path) -> list[Path]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.load(open(index, encoding="utf-8"))["weight_map"].values()))
        return [src / n for n in names]
    single = src / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(src.glob("model-*-of-*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards in {src}")
    return shards


def collect(src: Path) -> list[tuple[str, Path, dict, int]]:
    """All tensors of the directory: (name, file, entry, data_start), sorted by name."""
    tensors = []
    for shard in shard_files(src):
        if shard.suffix == ".part" or not shard.exists():
            raise SystemExit(f"shard missing or incomplete: {shard}")
        header, start = read_header(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry["dtype"] not in DTYPE_BYTES:
                raise SystemExit(f"{shard}: unsupported dtype {entry['dtype']} for {name}")
            tensors.append((name, shard, entry, start))
    tensors.sort(key=lambda t: t[0])
    seen = set()
    for name, *_ in tensors:
        if name in seen:
            raise SystemExit(f"duplicate tensor {name}")
        seen.add(name)
    return tensors


def write_merged(tensors, out: Path, metadata: dict, keep_dtype: bool, chunk_elements: int = 1 << 24):
    """Streams every tensor into one safetensors file, casting F32 -> BF16 unless keep_dtype."""
    entries = {}
    offset = 0
    for name, _shard, entry, _start in tensors:
        dtype = entry["dtype"]
        if dtype == "F32" and not keep_dtype:
            dtype = "BF16"
        if dtype == "F16":
            raise SystemExit(f"{name}: F16 sources are not supported (gufo reads F32 / BF16)")
        count = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 1
        size = count * DTYPE_BYTES[dtype]
        entries[name] = {"dtype": dtype, "shape": entry["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    header = {"__metadata__": metadata, **entries}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    total = len(tensors)
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for i, (name, shard, entry, start) in enumerate(tensors, 1):
            begin, end = entry["data_offsets"]
            source_dtype = entry["dtype"]
            target_dtype = entries[name]["dtype"]
            with open(shard, "rb") as src_file:
                src_file.seek(start + begin)
                remaining = end - begin
                element = DTYPE_BYTES[source_dtype]
                while remaining > 0:
                    take = min(remaining, chunk_elements * element)
                    raw = src_file.read(take)
                    if len(raw) != take:
                        raise SystemExit(f"short read in {shard} for {name}")
                    if source_dtype == "F32" and target_dtype == "BF16":
                        block = np.frombuffer(raw, dtype=np.float32)
                        f.write(bf16_from_f32(block).tobytes())
                    else:
                        f.write(raw)
                    remaining -= take
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} tensors", file=sys.stderr, flush=True)
    os.replace(tmp, out)


def dit_variant(src: Path) -> str:
    config = src / "config.json"
    if config.exists():
        cfg = json.load(open(config, encoding="utf-8"))
        version = str(cfg.get("model_version", "")).lower()
        if version in ("turbo", "sft", "base"):
            return version
        if cfg.get("is_turbo"):
            return "turbo"
    name = src.name.lower()
    for v in ("turbo", "sft", "base"):
        if v in name:
            return v
    raise SystemExit(f"cannot tell the variant of {src}: pass a directory with config.json")


def check_dit(tensors, src: Path) -> None:
    names = {t[0] for t in tensors}
    required = [
        "decoder.layers.31.mlp.down_proj.weight",
        "decoder.condition_embedder.weight",
        "encoder.text_projector.weight",
        "encoder.lyric_encoder.norm.weight",
        "encoder.timbre_encoder.special_token",
        "tokenizer.quantizer.project_in.weight",
        "tokenizer.quantizer.project_out.weight",
        "tokenizer.attention_pooler.special_token",
        "detokenizer.proj_out.weight",
        "null_condition_emb",
    ]
    missing = [n for n in required if n not in names]
    if missing:
        raise SystemExit(f"{src}: not an ACE-Step 1.5 XL checkpoint, missing {missing}")
    config = src / "config.json"
    if config.exists():
        cfg = json.load(open(config, encoding="utf-8"))
        expect = {"hidden_size": 2560, "num_hidden_layers": 32, "encoder_hidden_size": 2048,
                  "intermediate_size": 9728, "num_attention_heads": 32, "num_key_value_heads": 8}
        bad = {k: cfg.get(k) for k, v in expect.items() if cfg.get(k) != v}
        if bad:
            raise SystemExit(f"{src}: not the XL geometry gufo implements: {bad}")
    index = src / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.load(open(index, encoding="utf-8"))["weight_map"])
        if expected != names:
            raise SystemExit(f"{src}: tensor set differs from the index "
                             f"(missing {sorted(expected - names)[:5]}, extra {sorted(names - expected)[:5]})")


def lm_size(tensors, src: Path) -> str:
    by_name = {t[0]: t[2] for t in tensors}
    embed = by_name.get("model.embed_tokens.weight")
    if embed is None or len(embed["shape"]) != 2:
        raise SystemExit(f"{src}: no model.embed_tokens.weight; not a Qwen3 LM")
    vocab, width = embed["shape"]
    if vocab < 151669 + 64000:
        raise SystemExit(f"{src}: vocabulary {vocab} has no room for the 64000 audio codes")
    layers = 0
    while f"model.layers.{layers}.input_layernorm.weight" in by_name:
        layers += 1
    sizes = {1024: "0.6B", 2048: "1.7B", 2560: "4B"}
    size = sizes.get(width, f"{width}w")
    print(f"  LM: width {width}, {layers} layers, vocab {vocab} -> {size}"
          f"{' (untied lm_head)' if 'lm_head.weight' in by_name else ' (tied embeddings)'}",
          file=sys.stderr)
    return size


def convert_silence(src: Path, out: Path, force: bool) -> None:
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit("silence_latent.pt needs torch to read; the existing "
                         "silence_latent.safetensors works for every variant") from exc
    tensor = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        tensor = next(iter(tensor.values()))
    array = tensor.float().reshape(-1, 64).contiguous().numpy()
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (use --force)")
    header = {"__metadata__": {"format": "pt", "gufo.source": src.name},
              "silence_latent": {"dtype": "F32", "shape": list(array.shape),
                                 "data_offsets": [0, array.nbytes]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(array.astype("<f4").tobytes())
    print(f"wrote {out} [{array.shape[0]}, 64]", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["dit", "lm", "silence"])
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--keep-dtype", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.kind == "silence":
        convert_silence(args.src, args.out, args.force)
        return
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists (use --force)")
    tensors = collect(args.src)
    metadata = {"format": "pt", "gufo.source": args.src.name}
    if args.kind == "dit":
        check_dit(tensors, args.src)
        variant = dit_variant(args.src)
        metadata["gufo.model"] = f"ace-step-1.5-xl-{variant}"
        metadata["gufo.variant"] = variant
        metadata["model_version"] = variant
        print(f"DiT {variant}: {len(tensors)} tensors", file=sys.stderr)
    else:
        # acestep-5Hz-lm-1.7B ships its Qwen3 tensors without the "model." prefix
        # (embed_tokens.weight, layers.N.*); the 0.6B and 4B have it. Normalise.
        tensors = [((n if n.startswith(("model.", "lm_head.")) else "model." + n), s, e, o)
                   for n, s, e, o in tensors]
        size = lm_size(tensors, args.src)
        metadata["gufo.model"] = f"ace-step-5hz-lm-{size.lower()}"
        metadata["gufo.lm"] = size
        config = args.src / "config.json"
        if config.exists():
            cfg = json.load(open(config, encoding="utf-8"))
            metadata["tie_word_embeddings"] = str(cfg.get("tie_word_embeddings", True)).lower()
    total = sum(int(np.prod(t[2]["shape"], dtype=np.int64)) for t in tensors)
    print(f"{total / 1e9:.2f} B parameters -> {args.out}", file=sys.stderr)
    write_merged(tensors, args.out, metadata, args.keep_dtype)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)


if __name__ == "__main__":
    main()
