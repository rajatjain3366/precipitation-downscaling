# Smoke test only — not results

Produced by `python experiments/run_india.py --config configs/india_smoke.json --stage all`
(and `experiments/india_day_offset.py`) on `data/India_sample/`: **31 days of January 1990**
(train 1–17, val 18–24, test 25–31), a tiny network (channels 4/8/8, dense 8), 3 intervals,
2 epochs per model.

Every model is still at its initial output (about 0.5 mm/day everywhere), so QRE, BGNet and
all λ values give the same numbers to several decimals, and the λ "selection" is a tie. These
files only show that every stage runs end to end and writes its outputs. Do not quote them.
