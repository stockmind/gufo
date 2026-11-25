# ACE-Step 1.5 XL quality

`gufo-ace-check` (`src/models/ace_step/check_main.cpp`) compares the port stage
by stage with the torch oracle: the oracle writes raw FP32 and INT32 arrays plus
a `manifest.json`, and the checker reports cosine similarity and max abs error
for token ids, text embeddings, the conditioning tensor, the first velocity
sample and each sampler step.

```sh
gufo-ace-check --oracle DIR --case N --text text_encoder.safetensors \
    --dit dit.safetensors --tokenizer tokenizer.json --stages tokens,text,cond,v0,steps
```

No oracle run is recorded for this port yet. Audio quality therefore rests on
the 5 Hz planner and sampler matching the upstream task types, not on a source
comparison; record the per-stage cosines here when the oracle has been run on a
machine with `torch`.
