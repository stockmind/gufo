# ACE-Step 1.5 music

`gufo serve music` runs ACE-Step 1.5 XL (ACE Studio and StepFun, MIT) natively on HIP:
text and lyrics to 48 kHz stereo music, with the 5 Hz language-model planner and the
official task types (`text2music`, `repaint`, `cover`, `cover-nofsq`; `extract`, `lego`
and `complete` on the base variant). The request fields are listed in
[docs/internal/server.md](../internal/server.md) under `POST /v1/audio/music`.

One worker serves one DiT variant: turbo (8 steps, no guidance), sft (50 steps, APG
guidance 7) or base (32 steps, APG 7). Expose the three as three router sections.

## Files

`--model DIR` reads `dit.safetensors` (turbo), `text_encoder.safetensors`,
`vae.safetensors`, `tokenizer.json`, `silence_latent.safetensors` and, when present,
`lm.safetensors`; `--dit sft|base` picks `dit-sft.safetensors` / `dit-base.safetensors`,
`--lm none|0.6b|1.7b|4b` picks `lm-<size>.safetensors` (`lm.safetensors` is the 4B one).
The sampler variant comes from the DiT file's `gufo.variant` / `model_version` metadata.

`tools/ace_step/convert_hf_to_gufo.py dit HF_DIR DIR/dit-sft.safetensors` merges the
Hugging Face shards into one BF16 file with that metadata; `... lm HF_LM_DIR DIR/lm-1.7b.safetensors`
does the same for the planners (numpy only, no checkpoint code runs).

### Converted weights

The complete folder is published as
[D3rPapst/ace-step-1.5-gufo](https://huggingface.co/D3rPapst/ace-step-1.5-gufo) (44.9 GB;
MIT for the ACE-Step weights, Apache-2.0 for the Qwen3-Embedding-0.6B text encoder and
tokenizer; licence text included). Sources, converter commits and the three files taken
unchanged from the Comfy-Org repackaging are listed on the card.

```powershell
hf download D3rPapst/ace-step-1.5-gufo --local-dir D:\llm-modelle\gufo\modelle\ACE-Step\gufo
```

| File | Size | Content |
|---|---:|---|
| `dit.safetensors` | 9.97 GB | XL turbo DiT, BF16 |
| `dit-sft.safetensors` | 9.97 GB | XL sft DiT, BF16 |
| `dit-base.safetensors` | 9.97 GB | XL base DiT, BF16 |
| `lm.safetensors` | 8.38 GB | 5 Hz planner 4B, BF16 |
| `lm-1.7b.safetensors` | 3.71 GB | 5 Hz planner 1.7B, BF16 |
| `lm-0.6b.safetensors` | 1.33 GB | 5 Hz planner 0.6B, BF16 |
| `text_encoder.safetensors` | 1.19 GB | Qwen3-Embedding-0.6B text encoder, BF16 |
| `vae.safetensors` | 337 MB | Oobleck VAE (encoder + decoder), BF16 |
| `tokenizer.json` | 11.4 MB | Qwen3 tokenizer |
| `silence_latent.safetensors` | 3.8 MB | silence latent, F32 |

Only turbo is needed for text-to-music: `dit.safetensors`, `text_encoder.safetensors`,
`vae.safetensors`, `tokenizer.json`, `silence_latent.safetensors` and one planner
(`hf download ... --include "dit.safetensors" "text_encoder.safetensors" "vae.safetensors" "tokenizer.json" "silence_latent.safetensors" "lm-0.6b.safetensors"`).

## Router sections

From the production preset of the 64 GB machine (release 2.1.0):

```ini
[external/ace-step-1.5-xl-turbo]
command = D:\llm-modelle\gufo\app\gufo-2.1.0-2026-11-14\gufo.exe
args = serve music
model = D:\llm-modelle\gufo\modelle\ACE-Step\gufo
served-model-name = ace-step-1.5-xl-turbo

[external/ace-step-1.5-xl-sft]
command = D:\llm-modelle\gufo\app\gufo-2.1.0-2026-11-14\gufo.exe
args = serve music --dit sft --lm 1.7b
model = D:\llm-modelle\gufo\modelle\ACE-Step\gufo
served-model-name = ace-step-1.5-xl-sft
max-request-bytes = 67108864
router-gpu-gb = 15
router-host-gb = 1.5

[external/ace-step-1.5-xl-base]
command = D:\llm-modelle\gufo\app\gufo-2.1.0-2026-11-14\gufo.exe
args = serve music --dit base --lm 1.7b
model = D:\llm-modelle\gufo\modelle\ACE-Step\gufo
served-model-name = ace-step-1.5-xl-base
max-request-bytes = 67108864
router-gpu-gb = 15
router-host-gb = 1.5
```

## Measured

Strix Halo, 64 GB: turbo, 30 s of music, about 22 s with the 5 Hz planner (about 2 s
without); GPU 10.7 GB idle, about 15 GB peak; host RAM under 1 GB. sft and base run two
DiT passes per step (APG), so about twice the turbo time per step.
