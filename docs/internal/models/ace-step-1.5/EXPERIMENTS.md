# ACE-Step 1.5 XL experiments

| Experiment | Decision / evidence |
| --- | --- |
| One worker per DiT variant | Retained: turbo, sft and base differ in the 9.97 GB checkpoint, so a `--dit` switch would mean a reload; the router exposes three sections instead. |
| Planner at 5 Hz, optional | Retained: `--lm none` skips the 8.38 GB 4B planner and drops a 30 s turbo request from 22 s to 2 s, at the cost of the language-model plan and lyrics alignment. |
| Parallel safetensors read | Retained: 8 MiB pieces through `ReadFileAt` on one handle (`parallel_read.cpp`), which keeps the load off the critical path. |
| Preview of the clean estimate during sampling | Retained: the preview window decodes `x0 = x - t * v` per step and a decode failure only drops the preview, never the generation. |
| Two DiT passes per step for sft and base | Retained: APG guidance needs the conditional and unconditional velocity in the same step. |
| `gufo-ace-check` against a torch oracle | Open: the checker exists (`check_main.cpp`), no oracle run recorded yet; see [QUALITY](QUALITY.md). |
