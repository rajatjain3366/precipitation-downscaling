# End-term presentation

`BTP_endterm.pptx`: the end-term deck (25 slides plus 5 backup slides, 16:9, speaker notes on every slide).

Every number on the slides and in the notes comes from a committed file. [NUMBERS.md](NUMBERS.md) lists each one with its source file and field.

## Presenters

The speaker notes start with the presenter of each slide:

| Presenter | Slides | Part |
| --- | --- | --- |
| Raman Gupta | 2–12 | Motivation, paper, NZ replication, mid-term diagnosis, fixes |
| Rajat Jain | 13–21, 23–24 | India data, mass-conservation loss, protocol, final results, limitations, future work |
| Both | 1, 22, 25 and backup | Title, key findings, thank-you, backup slides |

## Rebuilding

From the repository root, with the project's Python environment:

```bash
python docs/presentation/figures/extract_numbers.py   # numbers.json from committed results, configs and docs
python docs/presentation/figures/make_figures.py      # charts (fig_*.png) and map crops (crop_*.png)
cd docs/presentation
npm install                                           # pptxgenjs and jszip, once
node build_deck.js                                    # BTP_endterm.pptx and NUMBERS.md
```

`export_with_powerpoint.ps1` (Windows, needs Microsoft PowerPoint) exports `BTP_endterm.pdf` and one PNG per slide to `render/`:

```powershell
powershell -ExecutionPolicy Bypass -File docs/presentation/export_with_powerpoint.ps1
```

The PDF and `render/` are not committed (generated files); `node_modules/` is not committed either.

## Files

| File | What it is |
| --- | --- |
| `figures/extract_numbers.py` | Reads the committed result files, configs, README and STATUS; writes `figures/numbers.json` (value, displayed text, source file, field). Text taken from markdown is checked to appear verbatim. |
| `figures/make_figures.py` | Charts from `numbers.json` only; the map crops cut the committed `example_maps.png` plots without redrawing. Same colour per model on every chart: BGNet slate, QRE amber, λ = 0 control light amber, QRE + mass teal. |
| `build_deck.js` | The deck (pptxgenjs). Every number goes through `n(key)`, which fails on a key missing from `numbers.json`. |
| `NUMBERS.md` | Generated: every number used, with its source. |
