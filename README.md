# qlora-eval

**Do LoRA and QLoRA actually improve a small model on a well-defined task — and what do they cost?**
An eval-first parameter-efficient fine-tuning kit: one task, one fixed eval set, three configs,
and numbers for quality, VRAM, and minutes.

> **Run it:** open [`qlora_eval.ipynb`](qlora_eval.ipynb) in Google Colab (T4 GPU) and
> *Runtime → Run all*. It downloads `Qwen/Qwen2.5-0.5B-Instruct`, trains both adapters, and
> writes `qlora_eval_results.json`.

## Protocol

| Item | Choice |
|---|---|
| Task | support-ticket triage → strict JSON `{"intent", "priority", "escalate"}` |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (fits a free Colab T4) |
| Data | synthetic, seeded: **400 train / 100 eval**, generated before training |
| Configs | zero-shot base · LoRA (fp16) · QLoRA (4-bit NF4) |
| Fixed across runs | seed, train steps (150), effective batch (8), LR (2e-4), eval set |
| Metrics | JSON validity, exact match, field accuracy, eval seconds, train seconds, peak VRAM |

- **Labels never leak:** the task JSON is generated deterministically from templates before any
  training; the eval set is disjoint from the train set.
- **Same budget:** both adapters train for 150 steps on the same data order and seed; LoRA rank 16,
  alpha 32, dropout 0.05 on `q_proj,k_proj,v_proj,o_proj`.
- **Cost is measured, not estimated:** training wall-clock and `torch.cuda.max_memory_allocated()`.

## Results

**Pending the first Colab run.** The notebook prints a table and saves `qlora_eval_results.json`;
paste the table here and commit the JSON under `results/`.

| config | JSON valid | exact match | field accuracy | eval s | train s | peak VRAM |
|---|---|---|---|---|---|---|
| base | — | — | — | — | — | — |
| lora | — | — | — | — | — | — |
| qlora | — | — | — | — | — | — |

## Files

- `qlora_eval.ipynb` — the whole experiment: dataset generation, prompt format, metrics,
  baseline eval, LoRA training, QLoRA training, results table.
- `results/` — committed `qlora_eval_results.json` from real runs (no cherry-picking).
- The notebook is deliberately dependency-light: `transformers`, `peft`, `accelerate`, `bitsandbytes`.

## Limitations and what I'd do next

- One small model and one synthetic task: the protocol is the deliverable, not a leaderboard claim.
- Baseline uses greedy decoding; a fairer comparison would report variance over 3 seeds.
- Next: push the winning adapter to the Hugging Face Hub with a model card, and add a
  human-written held-out set to check template overfitting.
- Next: measure cost in dollars for a rented T4 and a 24 GB GPU, so the LoRA↔QLoRA trade-off is
  priced, not just timed.

## License

MIT.
