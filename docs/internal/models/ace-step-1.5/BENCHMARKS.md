# ACE-Step 1.5 XL benchmarks

The worker loads models once; every number below is request time with the load
excluded.

## Strix Halo, 64 GB

gfx1151, release build, 30 s of music, fixed prompt and seed:

| Variant | Planner | Time | Memory |
| --- | --- | ---: | --- |
| turbo (8 steps) | 0.6B, 5 Hz | 22 s | 10.7 GB idle, about 15 GB peak, under 1 GB host |
| turbo (8 steps) | none | 2 s | same |

sft and base run two DiT passes per step for APG guidance, so they are not
comparable per step with turbo. Their end-to-end times on this machine are not
measured yet; each step reads the 9.97 GB BF16 DiT, so the step count dominates.
