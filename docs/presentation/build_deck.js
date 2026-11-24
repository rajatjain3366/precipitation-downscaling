/*
 * Builds docs/presentation/BTP_endterm.pptx.
 *
 * Every number on a slide or in the speaker notes comes from figures/numbers.json through n(key);
 * numbers.json is made only from committed files by figures/extract_numbers.py (see NUMBERS.md).
 *
 *   cd docs/presentation
 *   npm install                                   # pptxgenjs, once
 *   python figures/extract_numbers.py             # from the repository root: numbers.json
 *   python figures/make_figures.py                # charts
 *   node build_deck.js                            # the deck
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const FIG = path.join(HERE, "figures");
const NUM = JSON.parse(fs.readFileSync(path.join(FIG, "numbers.json"), "utf8"));
const USED = new Set();
function n(key) {
  if (!(key in NUM)) throw new Error(`number ${key} not in numbers.json`);
  USED.add(key);
  return NUM[key].display;
}
function pm(ds, model, metric) { return `${n(`${ds}.rep.${model}.${metric}.mean`)} ± ${n(`${ds}.rep.${model}.${metric}.std`)}`; }

// ------------------------------------------------------------------ design
const W = 13.333, H = 7.5, MX = 0.6;
const C = {
  navy: "0F2D4A", ink: "1F2A36", muted: "5F6B78", line: "D5DCE3", panel: "F2F5F8", white: "FFFFFF",
  teal: "167D7F", tealDark: "0E5557", tealLight: "E3F1F1", amber: "D9822B", amberLight: "F0C28A",
  slate: "5B6770", purple: "7A5195", red: "B23A48", redLight: "F6E3E6", ice: "BFD7EA",
};
const HEAD = "Cambria", BODY = "Calibri", MATH = "Cambria Math";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "QRE for Precipitation Downscaling: BTP end-term";
pres.subject = "BTP end-term evaluation";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: W - 2 * MX, h: 0.8, fontFace: HEAD,
      fontSize: 30, bold: true, color: C.ink, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Quantile-Regression-Ensemble for precipitation downscaling · BTP end-term",
      options: { x: MX, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: C.muted, fontFace: BODY, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.muted, fontFace: BODY, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: C.navy }, objects: [] });
["Introduction", "Replication", "Mid-term", "Fixes", "India", "Results", "Conclusion", "Backup"]
  .forEach((title) => pres.addSection({ title }));

// ------------------------------------------------------------------ helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function image(slide, file, box, name) {
  const p = path.join(FIG, file);
  const s = pngSize(p);
  const r = Math.min(box.w / s.w, box.h / s.h);
  const w = s.w * r, h = s.h * r;
  slide.addImage({ path: p, x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h, altText: name, objectName: name });
}
function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { x: MX, y: 0.3, w: 8, h: 0.3, fontSize: 12, bold: true, color: C.teal,
    charSpacing: 2, fontFace: BODY, margin: 0, isTextBox: true, objectName: "kicker" });
  s.addText(title, { placeholder: "title", align: "left" });
  return s;
}
function takeaway(s, text, y = 6.22) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: W - 2 * MX, h: 0.62, fill: { color: C.tealLight },
    line: { color: C.tealLight }, rectRadius: 0.08, objectName: "takeaway box" });
  s.addText(text, { x: MX + 0.25, y, w: W - 2 * MX - 0.5, h: 0.62, fontSize: 17, bold: true, color: C.tealDark,
    valign: "middle", fontFace: BODY, margin: 0, isTextBox: true, objectName: "takeaway" });
}
function source(s, text, y = 5.92) {
  s.addText(`Source: ${text}`, { x: MX, y, w: W - 2 * MX, h: 0.26, fontSize: 10, italic: true, color: C.muted,
    fontFace: BODY, margin: 0, isTextBox: true, objectName: "source" });
}
// a card: tinted rounded box with a bold heading and body text
function card(s, x, y, w, h, head, body, opt = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opt.fill || C.panel },
    line: { color: opt.fill || C.panel }, rectRadius: 0.08, objectName: `card ${head}` });
  const runs = [{ text: head, options: { bold: true, fontSize: opt.headSize || 16, color: opt.headColor || C.ink, breakLine: true } }];
  (Array.isArray(body) ? body : [body]).forEach((t, i, a) => runs.push({ text: t, options: {
    fontSize: opt.size || 15, color: C.ink, breakLine: i < a.length - 1, paraSpaceBefore: 4 } }));
  s.addText(runs, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: h - 0.2, valign: "top", fontFace: BODY, margin: 0,
    isTextBox: true, objectName: `card text ${head}` });
}
function stat(s, x, y, w, big, label, color) {
  s.addText([{ text: big, options: { fontSize: 34, bold: true, color: color || C.teal, breakLine: true } },
             { text: label, options: { fontSize: 14, color: C.muted } }],
    { x, y, w, h: 1.05, fontFace: BODY, margin: 0, valign: "top", isTextBox: true, objectName: `stat ${label}` });
}
function bullets(s, items, box, size = 18) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { ...box, fontSize: size, color: C.ink, fontFace: BODY, valign: "top", margin: 0, isTextBox: true, objectName: "bullets" });
}
function arrow(s, x1, y1, x2, y2, color = C.muted) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: 2, endArrowType: "triangle" }, objectName: "arrow" });
}
function table(s, rows, opt) {
  const hdr = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: opt.size || 14 } }));
  const body = rows.slice(1).map((r, i) => r.map((t, j) => {
    const cell = typeof t === "object" ? t : { text: t };
    return { text: cell.text, options: { fontSize: opt.size || 14, color: cell.color || C.ink, bold: cell.bold || j === 0,
      fill: { color: cell.fill || (i % 2 ? C.white : C.panel) } } };
  }));
  s.addTable([hdr, ...body], { x: opt.x, y: opt.y, w: opt.w, colW: opt.colW, fontFace: BODY, valign: "middle",
    border: { type: "solid", pt: 0.75, color: C.line }, margin: 0.06, rowH: opt.rowH || 0.42, autoPage: false,
    objectName: opt.name || "table" });
}
const A = "[Raman Gupta]", B = "[Rajat Jain]", AB = "[Raman Gupta and Rajat Jain]";
function notes(s, who, text) { s.addNotes(`${who} ${text.replace(/\s+/g, " ").trim()}`); }

// ================================================================== 1. title
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addText("B.TECH PROJECT · END-TERM EVALUATION", { x: MX, y: 0.75, w: 8, h: 0.35, fontSize: 14, bold: true,
    color: C.ice, charSpacing: 3, fontFace: BODY, margin: 0, isTextBox: true });
  s.addText("Quantile-Regression-Ensemble for Precipitation Downscaling:", { x: MX, y: 1.35, w: 7.9, h: 1.55,
    fontSize: 36, bold: true, color: C.white, fontFace: HEAD, margin: 0, valign: "top", isTextBox: true, objectName: "title" });
  s.addText("Replication, Fixes and an India Extension with Mass Conservation", { x: MX, y: 2.95, w: 7.9, h: 1.0,
    fontSize: 24, color: C.ice, fontFace: HEAD, margin: 0, valign: "top", isTextBox: true, objectName: "subtitle" });
  s.addText([
    { text: "Raman Gupta (S20240010200)", options: { bold: true, breakLine: true } },
    { text: "Rajat Jain (S20240010197)", options: { bold: true, breakLine: true } },
    { text: "Guide: Dr. Sumanta Chandra Mishra Sharma", options: { breakLine: true, paraSpaceBefore: 8 } },
    { text: "Indian Institute of Information Technology, Sri City, Chittoor" },
  ], { x: MX, y: 4.35, w: 7.9, h: 2.1, fontSize: 18, color: C.white, fontFace: BODY, margin: 0, valign: "top",
    paraSpaceAfter: 4, isTextBox: true, objectName: "team" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y: 1.05, w: 3.8, h: 4.45, fill: { color: C.white },
    line: { color: C.white }, rectRadius: 0.12, objectName: "map frame" });
  image(s, "crop_india_observed_map.png", { x: 9.1, y: 1.2, w: 3.5, h: 4.15 }, "Observed IMD rainfall, 2011-08-16");
  s.addText("Observed IMD rainfall on one test day (results/india/…/placements/plots)", { x: 8.95, y: 5.6, w: 3.8, h: 0.5,
    fontSize: 11, color: C.ice, fontFace: BODY, margin: 0, isTextBox: true });
  notes(s, AB, `Good morning. We are presenting our B.Tech project: a replication of the Quantile-Regression-Ensemble
    paper from AAAI 2024 for downscaling extreme precipitation, the problems we found and fixed in our own first
    implementation, and an extension to India with a mass-conservation loss. Raman covers the replication, the
    mid-term diagnosis and the fixes on the New Zealand data. Rajat covers the India extension, the
    mass-conservation loss and the final ten-repetition results. Every number on these slides comes from a committed
    result file in our repository; the speaker notes and a separate table list the file for each number.`);
}

// ================================================================== 2. motivation
{
  const s = content("Introduction", "Motivation", "Coarse models miss the rain that matters");
  bullets(s, [
    `Climate models and reanalyses give rainfall drivers on a coarse grid (here ${n("in.res")}); impacts need about ${n("in.imd_res")}.`,
    "Deep-learning downscaling trained on all days regresses to the mean and underestimates extreme rain.",
    "A downscaled field should also keep the total amount of water right: errors in totals add up.",
  ], { x: MX, y: 1.75, w: 6.2, h: 3.8 }, 19);
  card(s, 7.3, 1.7, 5.43, 3.6, "India, monsoon training days (IMD, JJAS 1971–2002)", [], { fill: C.panel });
  stat(s, 7.55, 2.35, 2.5, `${n("in.mean_rain")}`, "mm/day mean rainfall per cell", C.slate);
  stat(s, 10.15, 2.35, 2.5, `${n("in.p999")}`, "mm/day 99.9th percentile", C.amber);
  stat(s, 7.55, 3.85, 2.5, `${n("in.max_rain")}`, "mm/day largest value", C.red);
  stat(s, 10.15, 3.85, 2.5, `${n("in.zero_frac")}`, "of cell-days exactly dry", C.teal);
  source(s, "results/india/qre_mass_v1/data_check.json (imd/rain_train_years)");
  takeaway(s, "A downscaling model must get both the extremes and the totals right");
  notes(s, A, `Why downscaling? Global climate models and reanalyses describe the atmosphere on coarse grids, here
    about one degree, while impact studies need rainfall at about a quarter degree. Statistical, deep-learning
    downscaling learns that mapping. The problem the paper addresses is that a network trained on all days regresses
    to the mean: it predicts the common, moderate rain well and underestimates the extremes, which matter most for
    floods. The numbers on the right are from our India training data: the mean is ${n("in.mean_rain")} millimetres per
    day per cell, the 99.9th percentile ${n("in.p999")} and the maximum ${n("in.max_rain")}; ${n("in.zero_frac")} of
    cell-days are completely dry. So the distribution is extremely skewed. A second requirement, which motivates our
    extension, is that the downscaled field should keep the total amount of water right.`);
}

// ================================================================== 3. the paper
{
  const s = content("Introduction", "The paper · Bailie, Koh, Rampal & Gibson, AAAI-24", "QRE: one specialist member per rain-intensity interval");
  const shades = ["DCE8F2", "B7D0E5", "8FB4D3", "6895BE", "4677A6", "2B5A88"];
  // predictors
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 2.75, w: 1.85, h: 1.55, fill: { color: C.panel }, line: { color: C.slate, width: 1.25 }, rectRadius: 0.08 });
  s.addText([{ text: "Predictors x", options: { bold: true, breakLine: true } }, { text: "q, t, u, v, w at 850 hPa" }],
    { x: MX + 0.1, y: 2.8, w: 1.65, h: 1.45, fontSize: 15, color: C.ink, align: "center", valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
  // members
  for (let i = 0; i < 6; i++) {
    const y = 1.62 + i * 0.56;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.25, y, w: 2.55, h: 0.46, fill: { color: shades[i] }, line: { color: shades[i] }, rectRadius: 0.06 });
    s.addText(`f${String.fromCharCode(0x2081 + i)}  BGNet · interval ${i + 1}${i === 5 ? " (extremes)" : ""}`, { x: 3.3, y, w: 2.45, h: 0.46,
      fontSize: 13, bold: true, color: i >= 3 ? C.white : C.ink, valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
  }
  // weight network
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.25, y: 5.12, w: 2.55, h: 0.72, fill: { color: C.amberLight }, line: { color: C.amber, width: 1.25 }, rectRadius: 0.08 });
  s.addText([{ text: "Weight Network ω(x)", options: { bold: true, breakLine: true } }, { text: "softmax over the 6 intervals" }],
    { x: 3.3, y: 5.12, w: 2.45, h: 0.72, fontSize: 13, color: C.ink, align: "center", valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
  // sum and output
  s.addShape(pres.shapes.OVAL, { x: 6.5, y: 3.0, w: 1.25, h: 1.25, fill: { color: C.white }, line: { color: C.amber, width: 2 } });
  s.addText("Σ ωᵢ fᵢ", { x: 6.5, y: 3.0, w: 1.25, h: 1.25, fontSize: 18, bold: true, color: C.ink, align: "center", valign: "middle", fontFace: MATH, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y: 2.95, w: 1.55, h: 1.35, fill: { color: C.amber }, line: { color: C.amber }, rectRadius: 0.08 });
  s.addText([{ text: "g(x)", options: { bold: true, fontSize: 20, breakLine: true } }, { text: "fine-grid rainfall", options: { fontSize: 13 } }],
    { x: 8.4, y: 2.95, w: 1.45, h: 1.35, color: C.white, align: "center", valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
  arrow(s, 2.5, 3.2, 3.2, 2.6); arrow(s, 2.5, 3.85, 3.2, 5.3);
  arrow(s, 5.85, 3.25, 6.48, 3.45); arrow(s, 5.85, 5.4, 6.62, 4.12); arrow(s, 7.78, 3.62, 8.33, 3.62);
  s.addText("Training days are split into 6 intervals by their domain-total rain; member i trains only on interval i; "
    + "ω is trained (EMD loss) to predict the interval of a day.", { x: MX, y: 6.0, w: 9.3, h: 0.7, fontSize: 14,
    color: C.muted, fontFace: BODY, margin: 0, valign: "top", isTextBox: true });
  card(s, 10.15, 1.62, 2.58, 2.95, "BGNet (every member)", [
    "p = P(rain) per cell",
    "α, β = Gamma shape, scale",
    "Prediction ŷ = p·α·β",
    "Loss: Bernoulli–Gamma NLL",
    "Same network = the single-BGNet baseline",
  ], { size: 14 });
  notes(s, A, `This is the method from the paper by Bailie, Koh, Rampal and Gibson at AAAI 2024. The base network is
    BGNet. For every grid cell it predicts three things: the probability of rain p, and the shape alpha and scale beta of
    a Gamma distribution for the amount; it is trained with the Bernoulli-Gamma negative log-likelihood, and its point
    prediction is the expected value p times alpha times beta. QRE first splits the training days into six intervals by
    their domain-total rainfall. One BGNet member is trained only on the days of its interval, so the last member
    specialises in extremes. A separate Weight Network looks at the predictors and outputs softmax weights omega over
    the six intervals, trained with an Earth Mover's Distance loss. The final prediction is the weighted sum of the
    members, g of x equals the sum of omega i times f i. A single BGNet with the same architecture is the baseline.`);
}

// ================================================================== 4. replication setup
{
  const s = content("Replication", "Replication setup", "Our replication: CORDEX-ML-Bench, New Zealand");
  table(s, [
    ["", "Paper", "Ours (NZ)"],
    ["Predictors", n("paper.pred_shape"), `T, Q, U, V at 850 hPa, ${n("nz.x_shape")}`],
    ["Target", n("paper.target"), `regional-model rainfall, ${n("nz.target_res")}, ${n("nz.land_cells")} land cells (orog > 0)`],
    ["Split", n("paper.split"), `${n("nz.v2.train_years")} / ${n("nz.v2.val_years")} / ${n("nz.v2.test_years")} (${n("nz.v2.split_text")})`],
    ["Days", "–", `${n("nz.v2.train_days")} / ${n("nz.v2.val_days")} / ${n("nz.v2.test_days")}`],
    ["Conv kernel", "6", `${n("nz.kernel")} (${n("nz.grid16")} grid too small for six 6 × 6 convolutions)`],
  ], { x: MX, y: 1.65, w: 8.2, colW: [1.55, 2.9, 3.75], size: 14, rowH: 0.62 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.15, y: 1.65, w: 3.58, h: 3.9, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
  image(s, "crop_nz_observed_map.png", { x: 9.3, y: 1.75, w: 3.28, h: 3.45 }, "CORDEX NZ rainfall, 1996-07-16");
  s.addText("CORDEX rainfall on one NZ test day", { x: 9.15, y: 5.15, w: 3.58, h: 0.35, fontSize: 12, color: C.muted, align: "center", fontFace: BODY, margin: 0, isTextBox: true });
  source(s, "README.md §2 and §4; results/fixes_v2/dry_run/data_report.json; results/baseline_rerun/data_match_check.json");
  takeaway(s, "Same method on different data: absolute MSE values are not comparable with the paper's");
  notes(s, A, `The paper used ERA5 predictors and VCSN station-based observations, which were not available to us. On
    the advice of a co-author we used CORDEX-ML-Bench for New Zealand. The predictors are temperature, humidity and
    winds at 850 hectopascal on a 16 by 16 grid; vertical velocity is not in the benchmark. The target is regional
    climate model rainfall on a 128 by 128 grid, restricted to the ${n("nz.land_cells")} land cells where orography is
    above zero. We keep the paper's split proportions: ${n("nz.v2.train_years")} for training, ${n("nz.v2.val_years")}
    for validation and ${n("nz.v2.test_years")} for testing, that is ${n("nz.v2.train_days")}, ${n("nz.v2.val_days")}
    and ${n("nz.v2.test_days")} days. Because the grid is only 16 by 16, we use 3 by 3 convolutions instead of the
    paper's 6 by 6. So this is the same method on different data, and our absolute errors cannot be compared with the
    paper's tables.`);
}

// ================================================================== 5. mid-term results
{
  const s = content("Mid-term", "Mid-term · first implementation", "Mid-term: QRE was almost twice as bad as BGNet");
  image(s, "fig_midterm_results.png", { x: MX, y: 1.6, w: 8.6, h: 4.2 }, "Mid-term test MSE, QRE vs BGNet");
  card(s, 9.45, 1.65, 3.28, 3.3, "Mid-term run", [
    `Split ${n("mt.train_dates")} / ${n("mt.val_dates")} / ${n("mt.test_dates")} (${n("mt.train_years")} / ${n("mt.val_years")} / ${n("mt.test_years")} years)`,
    `Batch ${n("mt.batch")}, learning rate ${n("mt.lr")}, decay: ${n("mt.decay")}`,
    `At most ${n("mt.max_epochs_member")} epochs (members), ${n("mt.max_epochs_omega")} (Weight Network)`,
    `${n("mt.reps")} repetitions`,
  ], { size: 14 });
  source(s, "results/final_cordex_qre/metrics_table.csv, metadata.json (tag midterm-eval)");
  takeaway(s, `Test MSE [0, 1]: QRE ${n("mt.qre.0.0,1.0.mean_mse")} ± ${n("mt.qre.0.0,1.0.std_mse")} vs BGNet ${n("mt.single_bgnet.0.0,1.0.mean_mse")} ± ${n("mt.single_bgnet.0.0,1.0.std_mse")}: something was wrong`);
  notes(s, A, `At the mid-term we showed our first implementation. It used a split of ${n("mt.train_years")} training,
    ${n("mt.val_years")} validation and ${n("mt.test_years")} test years, batch size ${n("mt.batch")}, learning rate
    ${n("mt.lr")} without decay, and three repetitions. The result was the opposite of the paper's: on all test days the
    ensemble's mean squared error was ${n("mt.qre.0.0,1.0.mean_mse")}, against ${n("mt.single_bgnet.0.0,1.0.mean_mse")}
    for a single BGNet, and QRE was also worse on the driest 20 percent of days and on the wettest 10 percent, where it
    should help most: ${n("mt.qre.0.9,1.0.mean_mse")} against ${n("mt.single_bgnet.0.9,1.0.mean_mse")}. Its spread across
    repetitions was also large. Rather than reporting this as a property of the method, we audited the pipeline. The
    next three slides show what we found, with the evidence for each problem.`);
}

// ================================================================== 6. problem 1: WN collapse
{
  const s = content("Mid-term", "Diagnosis · problem 1", "Problem 1: the Weight Network collapsed");
  image(s, "fig_wn_collapse.png", { x: MX, y: 1.6, w: 6.9, h: 4.2 }, "Weight Network argmax histogram");
  card(s, 7.8, 1.65, 4.93, 1.55, "Symptom in the mid-term run", [
    `Best validation epoch of the Weight Network = ${n("mt.omega.rep0.best_epoch")} in all ${n("mt.reps")} repetitions (stopped after ${n("mt.omega.rep0.epochs")} epochs)`,
  ], { size: 15 });
  card(s, 7.8, 3.55, 4.93, 2.25, "Validation days, same EMD loss, after the other fixes", [
    `Argmax histogram ${n("s.wn.argmax")} (true ${n("s.wn.true")})`,
    `Mean max weight ${n("s.wn.mean_max")}; validation accuracy ${n("s.wn.accuracy")} < majority class ${n("wl.chance.majority")}`,
  ], { size: 15, fill: C.redLight });
  source(s, "results/final_cordex_qre/training_history/training_log.json; results/fixes_v2/nz/sanity/repetition_0.json");
  takeaway(s, "Every day went to one interval: QRE was just one member, never an ensemble");
  notes(s, A, `The first and most important problem was the Weight Network. In the mid-term run its best validation
    epoch was epoch ${n("mt.omega.rep0.best_epoch")} in every repetition, and early stopping ended it after
    ${n("mt.omega.rep0.epochs")} epochs: it learned nothing useful. After fixing everything else we measured it
    directly with the same EMD loss. The chart shows the result on the ${n("nz.v2.val_days")} validation days: the grey
    bars are the true intervals, spread over all six; the red bar is what the Weight Network chose: all of them in one
    interval. Its mean maximum weight is ${n("s.wn.mean_max")}, so it is completely confident, and its accuracy
    ${n("s.wn.accuracy")} is below simply always guessing the most common interval, ${n("wl.chance.majority")}. So the
    "ensemble" was really a single member applied to every day.`);
}

// ================================================================== 7. problem 2: inputs and baseline
{
  const s = content("Mid-term", "Diagnosis · problem 2", "Problem 2: unscaled inputs and an unfair baseline");
  image(s, "fig_predictor_scales.png", { x: MX, y: 1.6, w: 6.4, h: 4.2 }, "Raw predictor scales");
  card(s, 7.3, 1.65, 5.43, 1.3, "No input normalization", [
    `Training split: q850 has mean ${n("nz.raw_mean.q850")} kg/kg next to t850 at ${n("nz.raw_mean.t850")} K, fed to the network as they are`,
  ], { size: 15 });
  card(s, 7.3, 3.1, 5.43, 1.3, "Unfair baseline", [
    "The BGNet baseline had a different architecture from the members, so QRE vs BGNet was not like for like",
  ], { size: 15 });
  card(s, 7.3, 4.55, 5.43, 1.25, "Wrong prediction rule", [
    `Rain where p > 0.5, instead of the expected value p·α·β; rain threshold in the loss ${n("mt.problems.rain_threshold_old")}`,
  ], { size: 15 });
  source(s, "results/fixes_v2/dry_run/data_report.json (training-split statistics); README.md §5 and §6");
  takeaway(s, "Members learned from badly scaled inputs and were judged against a different model");
  notes(s, A, `The second group of problems concerns inputs and the comparison. The predictors were not normalized.
    The chart shows their raw training statistics on a log scale: specific humidity is around ${n("nz.raw_mean.q850")}
    kilograms per kilogram while temperature is around ${n("nz.raw_mean.t850")} kelvin, so without standardization the
    network sees inputs on completely different scales. Second, the single-BGNet baseline was built by different code,
    with a different architecture from the members, so the comparison was not like for like. Third, the prediction
    thresholded the rain probability at 0.5 instead of using the expected value p times alpha times beta that the
    paper uses, and the loss treated values up to 0.5 millimetres as dry; we now use 0.1.`);
}

// ================================================================== 8. problem 3: split, ties, GMM
{
  const s = content("Mid-term", "Diagnosis · problem 3", "Problem 3: split, ties and unstable intervals");
  image(s, "fig_gmm_midterm.png", { x: MX, y: 1.6, w: 7.3, h: 4.2 }, "Mid-term GMM intervals per repetition");
  card(s, 8.15, 1.65, 4.58, 1.25, "Split", [
    `${n("mt.train_years")} / ${n("mt.val_years")} / ${n("mt.test_years")} years, not the paper's ${n("paper.split")} proportions`,
  ], { size: 15 });
  card(s, 8.15, 3.05, 4.58, 1.3, "Ties counted twice", [
    `Bounds inclusive on both sides: segments held ${n("mt.gmm.rep0.sample_sum")} / ${n("mt.gmm.rep1.sample_sum")} days for ${n("mt.train_days")} training days`,
  ], { size: 15 });
  card(s, 8.15, 4.5, 4.58, 1.3, "Intervals refit per repetition", [
    "Bounds change between repetitions; reps 1 and 2 identical despite different seeds",
  ], { size: 15 });
  source(s, "results/final_cordex_qre/training_history/training_log.json (gmm_diagnostics); results/baseline_rerun/data_match_check.json");
  takeaway(s, "Members trained on different, overlapping segments in every repetition");
  notes(s, A, `The third group is about data handling. The split used ${n("mt.train_years")} years for training,
    ${n("mt.val_years")} for validation and ${n("mt.test_years")} for testing, far from the paper's 32 to 6 to 8
    proportions, so the members saw much less data. The interval bounds were inclusive on both sides, so days exactly
    on a bound were put into two segments: the six segments together held ${n("mt.gmm.rep0.sample_sum")} days in
    repetition 0, but there are only ${n("mt.train_days")} training days. Finally, the Gaussian mixture that defines the
    intervals was refitted in every repetition. The bars show the six intervals of each repetition: the bounds move
    between repetition 0 and the others, yet repetitions 1 and 2 gave identical bounds with different seeds. So each
    repetition trained its members on different, overlapping segments.`);
}

// ================================================================== 9. fixes
{
  const s = content("Fixes", "Fixes v2", "Fixes v2: every problem has a fix and a config flag");
  const fixes = [
    ["Split", `${n("nz.v2.train_years")} / ${n("nz.v2.val_years")} / ${n("nz.v2.test_years")} (${n("nz.v2.train_fraction")} / ${n("nz.v2.val_fraction")} / ${n("nz.v2.test_fraction")})`],
    ["Normalization", "Predictors standardized with training-split statistics only"],
    ["Prediction", `Expected value p·α·β; rain threshold ${n("cfg.nz.rain_threshold")} mm/day`],
    ["Fair baseline", "Members and BGNet built by one function (same architecture)"],
    ["Segmentation", "GMM fitted once; tᵢ ≤ total < tᵢ₊₁; labels use training thresholds"],
    ["Member validation", "Each member early-stops on its own interval's validation days"],
    ["Training", `Paper settings: LR decay ${n("cfg.nz.decay")} every ${n("cfg.nz.decay_steps")} steps, batch ${n("cfg.nz.batch")}`],
    ["Weight Network loss", "Squared EMD on CDFs (next slide)"],
  ];
  fixes.forEach(([h, b], i) => {
    const col = i % 4, row = Math.floor(i / 4);
    card(s, MX + col * 3.07, 1.65 + row * 2.12, 2.9, 1.95, h, b, { size: 15, fill: row === 1 && col === 3 ? C.tealLight : C.panel });
  });
  source(s, "README.md §4 and §6 (config flags); configs/cordex_v2.json");
  takeaway(s, "Each fix can be switched off with a flag in configs/cordex_v2.json to reproduce the old behaviour");
  notes(s, A, `Here are the fixes, implemented as a second version of the pipeline. We moved to the
    ${n("nz.v2.train_years")}, ${n("nz.v2.val_years")}, ${n("nz.v2.test_years")} split, which has the paper's 70, 12.5,
    17.5 percent proportions. Predictors are standardized using statistics from the training split only. Predictions
    use the expected value p alpha beta. Members and the baseline BGNet are built by the same function, so the
    comparison is fair. The intervals are fitted once with half-open bounds so every day belongs to exactly one segment,
    and validation and test labels use the training thresholds. Each member early-stops on the validation days of its
    own interval, and we use the paper's learning-rate decay. Every fix has a configuration flag that restores the old
    behaviour, which let us check each one separately. The last card, the Weight Network loss, needed its own study.`);
}

// ================================================================== 10. EMD collapse and loss study
{
  const s = content("Fixes", "Fixes · Weight Network loss", "Why the paper's EMD collapses, and what fixes it");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.6, w: 4.4, h: 4.2, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
  s.addText([
    { text: "Paper Eq. 6, linear in ω", options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true } },
    { text: "L = Σₖ |k − k′| ωₖ", options: { fontSize: 18, breakLine: true } },
    { text: "Minimum is one-hot; the softmax gradient vanishes there:", options: { fontFace: BODY, fontSize: 14, breakLine: true, paraSpaceBefore: 6 } },
    { text: "∂L/∂zₖ = ωₖ (cₖ − L) → 0", options: { fontSize: 18, breakLine: true } },
    { text: "Squared EMD on CDFs (Hou et al. 2016)", options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true, paraSpaceBefore: 12 } },
    { text: "Σₖ (CDF_ω(k) − CDF_y(k))²", options: { fontSize: 18, breakLine: true } },
    { text: "Smooth, nonzero gradient; still penalises distant intervals more.", options: { fontFace: BODY, fontSize: 14, paraSpaceBefore: 6 } },
  ], { x: MX + 0.2, y: 1.72, w: 4.0, h: 3.95, fontFace: MATH, color: C.ink, valign: "top", margin: 0, isTextBox: true, objectName: "loss formulas" });
  image(s, "fig_loss_study.png", { x: 5.2, y: 1.6, w: 7.53, h: 4.25 }, "Weight Network loss study");
  source(s, "results/fixes_v2/nz_wn/sanity/weight_network_losses/results.json (validation 1989–1993)");
  takeaway(s, `CE, squared EMD and CE + EMD all fix the collapse (validation accuracy ${n("wl.cross_entropy.accuracy")}–${n("wl.emd_squared_cdf.accuracy")}); we use squared EMD`);
  notes(s, A, `Why does the EMD loss collapse? The paper's loss is the sum over intervals of the distance to the true
    interval times the weight. It is linear in the weights, so its minimum is a one-hot vector, and the gradient
    through the softmax is omega k times c k minus L, which goes to zero once omega is one-hot. The network falls into
    one interval and cannot get out. We compared four losses on the same members. The paper's EMD gives accuracy
    ${n("wl.emd_paper.accuracy")}, below the majority-class rate of ${n("wl.chance.majority")}. Cross-entropy, squared EMD
    and their combination all reach about ${n("wl.cross_entropy.accuracy")} to ${n("wl.emd_squared_cdf.accuracy")}, and
    their QRE errors are within half a point of each other, which is within noise. We chose squared EMD because it is
    the EMD family the paper cites and is smooth. Note the purple line: even perfect weights give
    ${n("wl.reference_perfect_member.val.[0.0,1.0]")}, so the members themselves were the next bottleneck.`);
}

// ================================================================== 11. member study NZ
{
  const s = content("Fixes", "Fixes · member study (NZ)", "Members needed real training: warm start wins");
  image(s, "fig_member_study_nz.png", { x: MX, y: 1.6, w: 7.4, h: 4.2 }, "NZ member study own-interval ratios");
  card(s, 8.25, 1.65, 4.48, 2.0, "Perfect-Member validation MSE [0, 1]", [
    `10-epoch members: ${n("ms.sanity_10_epoch.pm_val")}`,
    `From scratch: ${n("ms.scratch.pm_val")}`,
    `Warm start from BGNet: ${n("ms.warm_start.pm_val")} (selected)`,
  ], { size: 15 });
  card(s, 8.25, 3.85, 4.48, 1.3, "Fair baseline: BGNet trained to convergence", [
    `Validation MSE [0, 1] ${n("ms.bgc_val")} (best epoch ${n("ms.bgc.best_epoch")} of ${n("ms.bgc.epochs")}); test ${n("ms.test.bgnet_converged.[0.0,1.0]")}`,
  ], { size: 15, fill: C.tealLight });
  source(s, "results/fixes_v2/nz_wn/sanity/member_study/results.json (validation 1989–1993)");
  takeaway(s, "With trained members, the best possible weighting reaches the level of a converged BGNet");
  notes(s, A, `The loss study showed that even perfect weights could not beat BGNet, because the members were trained
    for only ${n("cfg.nz.sanity_member_epochs")} epochs. So we ran a member study. The chart shows each member's error on
    its own interval divided by BGNet's; below one means the member wins. The grey bars are the 10-epoch members: they
    lose badly on the wet intervals. Trained to convergence, from scratch or warm-started from BGNet, every member
    beats the 10-epoch BGNet on its own interval. The Perfect-Member bound, the error with ideal weights, drops from
    ${n("ms.sanity_10_epoch.pm_val")} to ${n("ms.scratch.pm_val")} from scratch and ${n("ms.warm_start.pm_val")} with warm
    start, which we selected on validation. For a fair comparison, the baseline BGNet was then also trained to
    convergence: its validation error is ${n("ms.bgc_val")}.`);
}

// ================================================================== 12. NZ before vs after
{
  const s = content("Fixes", "NZ · before and after", "NZ: the fixes brought QRE close to BGNet");
  image(s, "fig_nz_before_after.png", { x: MX, y: 1.55, w: W - 2 * MX, h: 4.25 }, "NZ mid-term vs final test MSE");
  source(s, "results/final_cordex_qre/metrics_table.csv (test 1981–2000); results/fixes_v2/nz_wn/sanity/mass_study/repetitions/summary.json (test 1994–2000)");
  takeaway(s, `Test MSE [0, 1]: QRE ${n("mt.qre.0.0,1.0.mean_mse")} vs BGNet ${n("mt.single_bgnet.0.0,1.0.mean_mse")} at mid-term; ${n("nz.rep.qre.mse_0_1.mean")} vs ${n("nz.rep.bgnet.mse_0_1.mean")} now (different test years)`);
  notes(s, A, `This puts the mid-term and the final New Zealand results side by side for the three error measures. The
    hatched bars are the mid-term run, the solid bars the final ten-repetition run. One caution: the test periods
    differ, ${n("mt.test_dates")} at the mid-term and ${n("nz.v2.test_years")} now, so compare the gap between QRE and
    BGNet rather than absolute values. At the mid-term QRE's error was ${n("mt.qre.0.0,1.0.mean_mse")} against BGNet's
    ${n("mt.single_bgnet.0.0,1.0.mean_mse")}. After the fixes it is ${n("nz.rep.qre.mse_0_1.mean")} against
    ${n("nz.rep.bgnet.mse_0_1.mean")}, and the spread over repetitions is much smaller. On the driest days QRE is now on
    par with BGNet. So the fixes rescued QRE, but on this data it still does not beat a well-trained single BGNet;
    we come back to that with the statistics. That ends the New Zealand replication; Rajat continues with India.`);
}

// ================================================================== 13. India data
{
  const s = content("India", "India extension · data", "India: ERA5 → IMD, monsoon seasons 1971–2016");
  const rows = [
    ["Predictors", `ERA5 q, t, u, v, w at 850 hPa, daily means, 4 × 4 blocks → ${n("in.input_grid")} (${n("in.res")})`],
    ["Target", `IMD ${n("in.imd_res")} rainfall, ${n("in.imd_grid")} grid → ${n("in.cells")} cells`],
    ["Season", `JJAS: ${n("in.files.imd")} seasons × ${n("in.days_per_file")} days = ${n("in.n_days")} days`],
    ["Split", `${n("in.train_first")}–${n("in.train_last")} / ${n("in.val_first")}–${n("in.val_last")} / ${n("in.test_first")}–${n("in.test_last")}: ${n("in.train_days")} / ${n("in.val_days")} / ${n("in.test_days")} days`],
    ["Intervals", `Paper's fixed bounds: the GMM was unstable (${n("in.gmm.degenerate")})`],
    ["Day offset", `${n("in.offset.recommended")}: ERA5 day D → IMD day D (score difference ${n("in.offset.score_diff")}, train + val days)`],
  ];
  rows.forEach(([h, b], i) => {
    s.addText([{ text: h, options: { bold: true, color: C.teal, breakLine: true, fontSize: 14 } }, { text: b, options: { fontSize: 15, color: C.ink } }],
      { x: MX, y: 1.6 + i * 0.7, w: 6.35, h: 0.66, fontFace: BODY, margin: 0, valign: "top", isTextBox: true, objectName: `data row ${h}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.6, w: 2.4, h: 2.75, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
  image(s, "crop_india_observed_map.png", { x: 7.3, y: 1.67, w: 2.2, h: 2.6 }, "IMD rainfall map");
  image(s, "fig_india_intervals.png", { x: 9.75, y: 1.6, w: 2.98, h: 2.75 }, "India training days per interval");
  card(s, 7.2, 4.5, 5.53, 1.3, `${n("in.cells")} valid cells, ${n("in.area_total")} km²`, [
    `Paper's interval bounds ${n("paper.bounds")}; ${n("in.zero_frac")} of training cell-days dry`], { size: 14 });
  source(s, "results/india/qre_mass_v1/data_report.json, data_check.json, segmentation.json, day_offset/day_offset.json; docs/STATUS.md");
  takeaway(s, `The extremes member has only ${n("im.scratch.train_days.5")} training days (${n("in.top_member")})`);
  notes(s, B, `For India we built our own data. The predictors are ERA5 humidity, temperature, the two wind components
    and vertical velocity at 850 hectopascal, as daily means, averaged over 4 by 4 blocks to one degree, which gives a
    ${n("in.input_grid")} input grid. The target is the IMD quarter-degree gridded rainfall, ${n("in.cells")} cells over
    India. We use the monsoon season, June to September, for ${n("in.files.imd")} years, ${n("in.n_days")} days in total,
    split like the paper into 32, 6 and 8 years. The Bayesian mixture for the intervals was unstable on these totals,
    ${n("in.gmm.degenerate")}, so we use the paper's published interval bounds. That leaves the extremes member with only
    ${n("im.scratch.train_days.5")} training days, which will matter later. An empirical check chose day offset
    ${n("in.offset.recommended")}: ERA5 day D predicts IMD day D.`);
}

// ================================================================== 14. mass-conservation loss
{
  const s = content("India", "India extension · method", "Mass-conservation loss on the all-India daily total");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.6, w: 6.6, h: 3.75, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
  s.addText([
    { text: "Daily totals over valid cells (mask M, area A; observed y, predicted ŷ)", options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true } },
    { text: "Oₜ = Σᵢ Mᵢ Aᵢ yᵢ,ₜ        Sₜ = Σᵢ Mᵢ Aᵢ ŷᵢ,ₜ", options: { fontSize: 18, breakLine: true } },
    { text: "Absolute loss (area-mean mm/day, our default)", options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true, paraSpaceBefore: 10 } },
    { text: "L_mass = (1/B) Σₜ |Sₜ − Oₜ| / Σᵢ Mᵢ Aᵢ", options: { fontSize: 18, breakLine: true } },
    { text: `Cell areas, R = ${n("in.mass.R")}; valid cells total ${n("in.area_total")} km²`, options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true, paraSpaceBefore: 10 } },
    { text: "Aᵢ = R² Δλ (sin φ_north − sin φ_south)", options: { fontSize: 18, breakLine: true } },
    { text: "Combined, with λ scaled by the initial loss ratio", options: { bold: true, fontFace: BODY, fontSize: 15, breakLine: true, paraSpaceBefore: 10 } },
    { text: "L = L_QRE + λ · (L_QRE / L_mass)₀ · L_mass", options: { fontSize: 18 } },
  ], { x: MX + 0.25, y: 1.72, w: 6.1, h: 3.55, fontFace: MATH, color: C.ink, valign: "top", margin: 0, isTextBox: true, objectName: "formulas" });
  const pl = [["members", "Each member: NLL + λ·L_mass on its own days"],
              ["ensemble", "Fine-tune ω on g(x) = Σ ωᵢ fᵢ, members frozen"],
              ["both", "Both of the above, same λ"]];
  pl.forEach(([h, b], i) => card(s, 7.5, 1.6 + i * 0.98, 5.23, 0.86, `Placement "${h}"`, b, { size: 14, headSize: 15 }));
  card(s, 7.5, 4.55, 5.23, 1.25, "Selection (validation only)", [
    `λ ∈ {${n("in.study.lambdas")}} per placement; lowest val daily-total MAE among those within ${n("in.sel.tolerance")} val MSE of their own λ = 0 control`,
  ], { size: 14, fill: C.tealLight, headSize: 15 });
  source(s, "README.md §8; src/mass_conservation.py; configs/india_v1.json; results/india/qre_mass_v1/data_report.json");
  takeaway(s, "The control gets the same extra training with λ = 0, so the comparison isolates the mass term");
  notes(s, B, `Our extension adds a mass-conservation loss on the all-India daily total. For each day we sum rainfall
    times cell area over the valid cells, for the observation, O t, and for the prediction, S t. The loss is the
    absolute difference of the two totals, divided by the total area, so it is in area-mean millimetres per day; it is
    computed per day and averaged. Cell areas come from the spherical formula with an Earth radius of
    ${n("in.mass.R")}; the valid cells cover ${n("in.area_total")} square kilometres. We prefer the absolute form because
    a relative form would give near-dry days ten times the weight of monsoon days. The total loss adds lambda times
    L mass, with lambda scaled so that lambda equals one means both terms start equally large. The term can be placed
    in the members, in an ensemble fine-tuning stage, or both. Lambda and the placement are chosen on validation only,
    each against a lambda equals zero control that gets exactly the same extra training.`);
}

// ================================================================== 15. placement and lambda selection
{
  const s = content("India", "Method · validation-only selection", "Choosing placement and λ on validation");
  image(s, "fig_placements.png", { x: MX, y: 1.55, w: W - 2 * MX, h: 4.0 }, "Placement and lambda selection");
  s.addText([
    { text: `NZ: ensemble, λ = ${n("nz.sel.lambda")}: `, options: { bold: true } },
    { text: `val daily-total MAE ${n("nz.cand.ensemble.10.mae")} vs control ${n("nz.cand.ensemble.10.control_mae")} mm/day.   ` },
    { text: `India: members, λ = ${n("in.sel.lambda")}: `, options: { bold: true } },
    { text: `${n("in.cand.members.0.1.mae")} vs ${n("in.cand.members.0.1.control_mae")} mm/day.` },
  ], { x: MX, y: 5.6, w: W - 2 * MX, h: 0.35, fontSize: 14, color: C.ink, fontFace: BODY, margin: 0, isTextBox: true });
  source(s, "results/fixes_v2/nz_wn/sanity/mass_study/placements/selection.json; results/india/qre_mass_v1/placements/selection.json", 5.98);
  takeaway(s, `Training the members with the mass term costs NZ ${n("nz.sel.members_cost")} validation MSE; the ensemble placement costs nothing`, 6.3);
  notes(s, B, `Here is the validation sweep for both regions. Each line is a placement, and the y-axis is the
    validation error divided by that placement's own lambda equals zero control; the shaded band is the 2 percent
    tolerance. For New Zealand, putting the mass term into the members or both raises the error by 6 to 53 percent,
    so only the ensemble placement qualifies, and among those lambda equals ${n("nz.sel.lambda")} gives the lowest
    daily-total error, ${n("nz.cand.ensemble.10.mae")} against ${n("nz.cand.ensemble.10.control_mae")} for the control.
    For India the ensemble placement barely changes anything, while the members placement with lambda equals
    ${n("in.sel.lambda")} stays within tolerance and lowers the validation daily-total error from
    ${n("in.cand.members.0.1.control_mae")} to ${n("in.cand.members.0.1.mae")}. These choices were fixed before we
    looked at any test data.`);
}

// ================================================================== 16. protocol
{
  const s = content("Results", "Experimental protocol", "Protocol: one resumable pipeline, ten seeds, paired tests");
  const steps = [["NZ member study", "steps 1–3"], ["NZ placements", "step 11"], ["India data and member study", "steps 5–7, 10"],
                 ["India placements", "step 12"], ["10 repetitions", "steps 13–14"]];
  steps.forEach(([h, b], i) => {
    s.addShape(pres.shapes.CHEVRON, { x: MX + i * 2.45, y: 1.7, w: 2.55, h: 1.05, fill: { color: i === 4 ? C.teal : C.navy }, line: { color: C.white, width: 1 } });
    s.addText([{ text: h, options: { bold: true, breakLine: true, fontSize: 14 } }, { text: b, options: { fontSize: 12 } }],
      { x: MX + i * 2.45 + 0.42, y: 1.7, w: 1.75, h: 1.05, color: C.white, align: "center", valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
  });
  card(s, MX, 3.1, 3.9, 2.3, "Hardware and time", [
    `${n("run.gpu")} GPU, ${n("run.tf")}`,
    `Steps 0–9: ${n("run.first").replace(" (steps 0–9)", "")}`,
    `Steps 10–14: ${n("run.second").replace(" (steps 10–14)", "")}`,
  ], { size: 15 });
  card(s, 4.72, 3.1, 3.9, 2.3, "Repetitions", [
    `${n("in.study.reps")} repetitions, seed ${n("run.seeds")}`,
    "Every model retrained in each repetition: BGNet, members, Weight Network, mass fine-tune",
  ], { size: 15 });
  card(s, 8.84, 3.1, 3.89, 2.3, "Statistics", [
    "Paired two-sided Wilcoxon signed-rank test, α = 5 %",
    `With 10 pairs the smallest possible p is ${n("run.p_min")}: all 10 differences with one sign`,
  ], { size: 15, fill: C.tealLight });
  source(s, "docs/STATUS.md; results/lab_runs/2026-11-24_rtx4000ada/logs/ (gpu_check.log, summary.txt); scripts/run_lab.py");
  takeaway(s, "scripts/run_lab.py runs every step in order, skips finished steps and logs each run");
  notes(s, B, `All final experiments ran through one script, run lab, on the college lab PC with an
    ${n("run.gpu")} GPU. It runs the New Zealand member study, the placement studies and the repetitions in order,
    skips finished steps and keeps logs. The first ten steps took ${n("run.first").replace(" (steps 0–9)", "")}, the last
    five ${n("run.second").replace(" (steps 10–14)", "")}. For the final comparison we ran ${n("in.study.reps")} repetitions
    with different seeds, and in every repetition all models are retrained, so the spread includes the full training
    variation. We compare models with the paired two-sided Wilcoxon signed-rank test at 5 percent. With ten pairs the
    smallest p-value possible is ${n("run.p_min")}, which happens when all ten differences have the same sign; you will
    see that value often.`);
}

// ================================================================== 17. NZ final table
{
  const s = content("Results", "NZ · final results (10 repetitions, test 1994–2000)", "NZ: QRE trails a converged BGNet");
  const M = ["mse_0_1", "mse_0_0.2", "mse_0.9_1", "daily_total_mae", "abs_total_rel_discrepancy"];
  table(s, [
    ["Model (test)", "MSE [0, 1]", "MSE [0, 0.2]", "MSE [0.9, 1]", "Daily-total MAE", "|Total rel. discr.|"],
    ...[["bgnet", "BGNet (converged)"], ["qre", "QRE"], ["control", "QRE (λ = 0 control)"], ["qre_mass", `QRE + mass (ens., λ = ${n("nz.rep.lambda")})`]]
      .map(([m, l]) => [l, ...M.map((k) => pm("nz", m, k))]),
  ], { x: MX, y: 1.6, w: W - 2 * MX, colW: [2.85, 1.85, 1.85, 1.95, 1.8, 1.833], size: 14, name: "NZ results table" });
  const P = (t, k) => { const p = n(`nz.rep.p.${t}.${k}`); return { text: p, bold: Number(NUM[`nz.rep.p.${t}.${k}`].value) < 0.05, color: Number(NUM[`nz.rep.p.${t}.${k}`].value) < 0.05 ? C.tealDark : C.ink }; };
  table(s, [
    ["Wilcoxon p (test)", "MSE [0, 1]", "MSE [0, 0.2]", "MSE [0.9, 1]", "Daily-total MAE", "|Total rel. discr.|"],
    ...[["qre_vs_bgnet", "QRE vs BGNet"], ["qre_mass_vs_qre", "QRE + mass vs QRE"], ["qre_mass_vs_control", "QRE + mass vs control"]]
      .map(([t, l]) => [l, ...M.map((k) => P(t, k))]),
  ], { x: MX, y: 3.95, w: W - 2 * MX, colW: [2.85, 1.85, 1.85, 1.95, 1.8, 1.833], size: 14, name: "NZ p-value table" });
  source(s, "results/fixes_v2/nz_wn/sanity/mass_study/repetitions/summary.json and summary.md (mean ± std, n = 10); bold p < 0.05");
  takeaway(s, `QRE is worse than BGNet on test MSE [0, 1] and [0.9, 1] (p = ${n("nz.rep.p.qre_vs_bgnet.mse_0_1")}); totals are the same`);
  notes(s, B, `These are the final New Zealand numbers on the test years, mean and standard deviation over ten
    repetitions. The converged BGNet has an error of ${pm("nz", "bgnet", "mse_0_1")} on all days; QRE has
    ${pm("nz", "qre", "mse_0_1")}. The bottom table gives the Wilcoxon p-values; bold means significant. QRE is
    significantly worse than BGNet on all days and on the wettest 10 percent, p equals
    ${n("nz.rep.p.qre_vs_bgnet.mse_0_1")}, while on the driest days and on the totals there is no significant
    difference. The mass term in the ensemble with lambda ${n("nz.rep.lambda")} gives a small but consistent gain over
    its control: ${n("nz.rep.qre_mass.mse_0_1.mean")} against ${n("nz.rep.control.mse_0_1.mean")}, and daily-total error
    ${n("nz.rep.qre_mass.daily_total_mae.mean")} against ${n("nz.rep.control.daily_total_mae.mean")}, both with p equal to
    ${n("nz.rep.p.qre_mass_vs_control.mse_0_1")}.`);
}

// ================================================================== 18. NZ chart
{
  const s = content("Results", "NZ · final results", "NZ: small but consistent gains from the mass term");
  image(s, "fig_nz_final.png", { x: MX, y: 1.55, w: W - 2 * MX, h: 4.3 }, "NZ final results with error bars");
  source(s, "results/fixes_v2/nz_wn/sanity/mass_study/repetitions/summary.json (bars: mean ± std; dots: the 10 repetitions)");
  takeaway(s, `Test, vs its control: MSE [0, 1] ${n("nz.rep.qre_mass.mse_0_1.mean")} vs ${n("nz.rep.control.mse_0_1.mean")}, daily-total MAE ${n("nz.rep.qre_mass.daily_total_mae.mean")} vs ${n("nz.rep.control.daily_total_mae.mean")} (both p = ${n("nz.rep.p.qre_mass_vs_control.mse_0_1")})`);
  notes(s, B, `The same results as a chart. Bars are means with standard deviations, and each dot is one repetition.
    The gap between BGNet and the three QRE variants on all days and on extremes is clearly larger than the spread
    over repetitions. The light amber bar is the lambda equals zero control: the same ensemble fine-tuning as QRE plus
    mass, with the mass term switched off. Comparing the teal bar with it isolates the effect of the mass term. On all
    three panels the mass term moves the result in the right direction, and on the overall error and the daily-total
    error all ten repetitions agree, which is why p is ${n("nz.rep.p.qre_mass_vs_control.mse_0_1")}. The effect is small:
    the mass term cannot close the gap to BGNet.`);
}

// ================================================================== 19. India member study
{
  const s = content("Results", "India · member study (validation)", "India: too few days per member to train from scratch");
  image(s, "fig_india_member.png", { x: MX, y: 1.6, w: 7.4, h: 4.2 }, "India member study");
  card(s, 8.25, 1.65, 4.48, 1.95, "Perfect-Member validation MSE [0, 1]", [
    `From scratch: ${n("im.scratch.pm_val")}`,
    `Warm start: ${n("im.warm_start.pm_val")} (selected)`,
    `BGNet itself (validation): ${n("im.bgnet_val")}`,
  ], { size: 15 });
  card(s, 8.25, 3.8, 4.48, 1.75, "Training", [
    `Scratch: best epochs ${n("im.scratch.best_epoch.1")}–${n("im.scratch.best_epoch.2")}, two still improving at the ${n("cfg.in.max_epochs")}-epoch cap`,
    `Warm start: best epochs ${n("im.warm_start.best_epoch.0")}–${n("im.warm_start.best_epoch.3")}`,
  ], { size: 14 });
  source(s, "results/india/qre_mass_v1/member_study/results.json and selection.json");
  takeaway(s, `Members trained from scratch on ${n("im.scratch.train_days.5")}–${n("im.scratch.train_days.1")} training days each lose to BGNet on every interval (validation)`);
  notes(s, B, `For India we repeated the member study. Each member trained from scratch sees only the days of its own
    interval, between ${n("im.scratch.train_days.5")} and ${n("im.scratch.train_days.1")} days, the numbers at the bottom
    of the bars. Every scratch member is worse than BGNet on its own interval, by up to a third on the extremes
    interval, and two were still improving when they hit the epoch cap. Warm-started members begin from BGNet and stay
    within 2 percent of it. On the Perfect-Member error, warm start reaches ${n("im.warm_start.pm_val")} against
    ${n("im.scratch.pm_val")} from scratch, slightly below BGNet's ${n("im.bgnet_val")}, so warm start was selected. The
    lesson is that specialisation needs data: with 46 monsoon seasons, splitting the days six ways leaves too little for
    each member.`);
}

// ================================================================== 20. India final table
{
  const s = content("Results", "India · final results (10 repetitions, test 2009–2016 JJAS)", "India: QRE slightly beats BGNet; the mass term fixes totals");
  const M = ["mse_0_1", "mse_0_0.2", "mse_0.9_1", "daily_total_mae", "abs_total_rel_discrepancy"];
  table(s, [
    ["Model (test)", "MSE [0, 1]", "MSE [0, 0.2]", "MSE [0.9, 1]", "Daily-total MAE", "|Total rel. discr.|"],
    ...[["bgnet", "BGNet"], ["qre", "QRE (= λ = 0 control)"], ["qre_mass", `QRE + mass (members, λ = ${n("in.rep.lambda")})`]]
      .map(([m, l]) => [l, ...M.map((k) => pm("in", m, k))]),
  ], { x: MX, y: 1.6, w: W - 2 * MX, colW: [3.15, 1.8, 1.8, 1.85, 1.75, 1.783], size: 14, name: "India results table" });
  const P = (t, k) => { const v = Number(NUM[`in.rep.p.${t}.${k}`].value); return { text: n(`in.rep.p.${t}.${k}`), bold: v < 0.05, color: v < 0.05 ? C.tealDark : C.ink }; };
  table(s, [
    ["Wilcoxon p (test)", "MSE [0, 1]", "MSE [0, 0.2]", "MSE [0.9, 1]", "Daily-total MAE", "|Total rel. discr.|"],
    ...[["qre_vs_bgnet", "QRE vs BGNet"], ["qre_mass_vs_qre", "QRE + mass vs QRE"]].map(([t, l]) => [l, ...M.map((k) => P(t, k))]),
  ], { x: MX, y: 3.55, w: W - 2 * MX, colW: [3.15, 1.8, 1.8, 1.85, 1.75, 1.783], size: 14, name: "India p-value table" });
  s.addText("For the members placement the λ = 0 control is the QRE itself (no extra stage).", { x: MX, y: 4.95, w: W - 2 * MX, h: 0.35,
    fontSize: 14, color: C.muted, fontFace: BODY, margin: 0, isTextBox: true });
  source(s, "results/india/qre_mass_v1/repetitions/summary.json and summary.md (mean ± std, n = 10); bold p < 0.05");
  takeaway(s, `Test |total rel. discrepancy| ${n("in.rep.qre.abs_total_rel_discrepancy.mean")} → ${n("in.rep.qre_mass.abs_total_rel_discrepancy.mean")} (p = ${n("in.rep.p.qre_mass_vs_qre.abs_total_rel_discrepancy")}), at a cost in MSE`);
  notes(s, B, `The final India results. Here QRE beats BGNet, slightly but consistently: on all days
    ${pm("in", "qre", "mse_0_1")} against ${pm("in", "bgnet", "mse_0_1")}, p ${n("in.rep.p.qre_vs_bgnet.mse_0_1")}, and it
    is also significantly better on dry days, on extremes and on the daily-total error. The gain is small compared with
    the spread over repetitions, but all ten repetitions agree. The mass term, in the members with lambda
    ${n("in.rep.lambda")}, does what it is designed for: the absolute error in the total over the test period falls from
    ${n("in.rep.qre.abs_total_rel_discrepancy.mean")} to ${n("in.rep.qre_mass.abs_total_rel_discrepancy.mean")}, p
    ${n("in.rep.p.qre_mass_vs_qre.abs_total_rel_discrepancy")}. But the squared errors get worse, on all days and on the
    extremes. The daily-total error does not change.`);
}

// ================================================================== 21. India bias and cost
{
  const s = content("Results", "India · water balance vs accuracy", "Mass conservation removes India's rainfall deficit");
  image(s, "fig_india_bias.png", { x: MX, y: 1.6, w: 6.6, h: 4.25 }, "India signed total discrepancy");
  stat(s, 7.55, 1.75, 5.2, n("rep.in.bias_change"), "test: mean signed bias of the all-India total, QRE → QRE + mass", C.teal);
  stat(s, 7.55, 3.05, 2.5, `+${n("rep.in.mse_cost").split("+")[1].split(" ")[0]} %`, "test MSE [0, 1]", C.red);
  stat(s, 10.15, 3.05, 2.5, `+${n("rep.in.mse_cost").split("+")[2].split(" ")[0]} %`, "test MSE [0.9, 1]", C.red);
  card(s, 7.55, 4.3, 5.18, 1.5, "All three changes are significant", [
    `Test: p = ${n("in.rep.p.qre_mass_vs_qre.abs_total_rel_discrepancy")} for the totals and for both MSE values (all 10 repetitions agree)`,
  ], { size: 15, fill: C.tealLight });
  source(s, "docs/STATUS.md (signed discrepancy from the ten rep_<k>.json files); results/india/qre_mass_v1/repetitions/summary.json");
  takeaway(s, "The mass term fixes the water balance at a small cost in accuracy");
  notes(s, B, `This chart shows the signed error of the all-India rainfall total over the test period, for each model
    and each repetition. BGNet and QRE both under-predict monsoon rainfall, by about 3 percent on average:
    ${n("rep.signed.in.bgnet")} and ${n("rep.signed.in.qre")}. With the mass term the bias moves to
    ${n("rep.signed.in.mass")}, essentially zero, and the spread across repetitions also shrinks. That is the water
    balance fixed. The price is accuracy at the level of individual cells and days: the mean squared error rises by
    ${n("rep.in.mse_cost").split(" (")[0]} on all days and by ${n("rep.in.mse_cost").split(", ")[1].split(" (")[0]} on the extremes. All three changes are
    significant with p equal to ${n("in.rep.p.qre_mass_vs_qre.abs_total_rel_discrepancy")}. So the mass term trades a
    little point accuracy for correct totals; whether that is worth it depends on the application, for example water
    budgets versus local flood warnings.`);
}

// ================================================================== 22. key findings
{
  const s = content("Conclusion", "Conclusion", "Key findings");
  const F = [
    ["The fixes rescued QRE", `NZ test MSE [0, 1]: ${n("mt.qre.0.0,1.0.mean_mse")} (BGNet ${n("mt.single_bgnet.0.0,1.0.mean_mse")}) at mid-term → ${n("nz.rep.qre.mse_0_1.mean")} (BGNet ${n("nz.rep.bgnet.mse_0_1.mean")}); different test years`],
    ["QRE vs BGNet depends on the data", `Test MSE [0, 1]: NZ QRE worse (${n("nz.rep.qre.mse_0_1.mean")} vs ${n("nz.rep.bgnet.mse_0_1.mean")}); India: QRE slightly better (${n("in.rep.qre.mse_0_1.mean")} vs ${n("in.rep.bgnet.mse_0_1.mean")}); both p = ${n("nz.rep.p.qre_vs_bgnet.mse_0_1")}`],
    ["Mass conservation fixes the water balance", `India test total bias ${n("rep.in.bias_change")} at ${n("rep.in.mse_cost")}; NZ: small gains over its control`],
    ["Specialization needs data per member", `India scratch members (${n("im.scratch.train_days.5")}–${n("im.scratch.train_days.1")} training days) lose to BGNet on validation; warm start helps`],
    ["Evaluate fairly", "Same builder, converged baseline, λ = 0 control with the same training, 10 seeds, paired tests"],
  ];
  F.forEach(([h, b], i) => {
    const y = 1.6 + i * 0.98;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.12, w: 0.62, h: 0.62, fill: { color: i === 2 ? C.teal : C.navy }, line: { color: C.white } });
    s.addText(String(i + 1), { x: MX, y: y + 0.12, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", fontFace: BODY, margin: 0, isTextBox: true });
    s.addText([{ text: h, options: { bold: true, fontSize: 18, breakLine: true } }, { text: b, options: { fontSize: 15, color: C.muted } }],
      { x: MX + 0.85, y, w: W - 2 * MX - 0.85, h: 0.9, fontFace: BODY, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
  });
  notes(s, AB, `To summarise. First, the mid-term failure was caused by our implementation, not the method: after the
    fixes QRE went from nearly twice BGNet's error to close to it. Second, whether QRE beats a single BGNet depends on
    the data: on New Zealand a converged BGNet is better, on India QRE is slightly better, both with all ten repetitions
    agreeing. Third, the mass-conservation loss does what it is designed for: on India it removes the 3 percent rainfall
    deficit, at a cost of about 2 percent in squared error; on New Zealand it gives small gains over its control.
    Fourth, specialization needs data: members trained from scratch on a few hundred days each cannot beat a network
    trained on all days. And finally, a fair evaluation was essential: the same architecture, a converged baseline, a
    control with the same training, ten seeds and paired tests changed several of our conclusions.`);
}

// ================================================================== 23. limitations
{
  const s = content("Conclusion", "Honest notes", "Limitations");
  const L = [
    ["Data", "One dataset per region; the NZ targets are regional-climate-model output, not observations"],
    ["India setup", "Monsoon season (JJAS) only; no elevation input; predictors at 1°"],
    ["Deviations from the paper", "NZ kernel 3 (16 × 16 grid), squared EMD instead of Eq. 6, fixed India interval bounds"],
    ["Statistics", `${n("in.study.reps")} repetitions (the paper used ${n("paper.reps")}); one train / validation / test split per region`],
    ["Physics", "The mass term constrains only the domain total; no moisture flux or other physics"],
    ["Reproducibility", "GPU runs are not bit-identical to CPU runs (cuDNN kernel choice)"],
  ];
  L.forEach(([h, b], i) => card(s, MX + (i % 2) * 6.15, 1.6 + Math.floor(i / 2) * 1.45, 5.98, 1.3, h, b, { size: 15 }));
  source(s, "README.md §4, §8, §10; docs/STATUS.md; docs/LAB_SETUP.md §5");
  notes(s, B, `Some honest notes on the limits of this work. We have one dataset per region, and the New Zealand targets
    are output of a regional climate model, not observations. The India experiment covers only the monsoon season,
    without an elevation input, with one-degree predictors. We deviate from the paper in documented ways: a smaller
    kernel on the small New Zealand grid, the squared EMD loss, and fixed interval bounds for India. We ran ten
    repetitions instead of the paper's forty, on a single split per region. The mass term only constrains the total
    over the whole domain; it is a statistical constraint, not physics. And GPU results are not bit-identical to CPU
    results, so exact numbers can differ slightly between machines.`);
}

// ================================================================== 24. future work
{
  const s = content("Conclusion", "Next steps", "Future work");
  const F = [
    ["More data per member", "Shared backbone or multi-task members, fewer intervals, or augmentation for the extremes member"],
    ["Regional mass constraints", "Conserve totals per river basin or region, not only all of India"],
    ["Richer predictors", "Elevation, more pressure levels and full-year data for India"],
    ["Robust evaluation", "Several splits (e.g. leave-years-out), more repetitions, other GCMs for NZ"],
  ];
  F.forEach(([h, b], i) => card(s, MX + (i % 2) * 6.15, 1.75 + Math.floor(i / 2) * 1.95, 5.98, 1.6, h, b, { size: 16, headSize: 18, fill: i === 1 ? C.tealLight : C.panel }));
  notes(s, B, `Where would we go next? The clearest lesson is that members need more data: sharing a backbone between
    members, multi-task training, fewer intervals, or augmentation for the extremes member could let specialization
    pay off. For the mass term, a natural step is to conserve totals per region or river basin instead of one all-India
    total, which matters more for hydrology. For India we would add elevation, more pressure levels and the full year.
    And for evaluation, several splits and more repetitions, and on New Zealand other global models, would make the
    conclusions more general.`);
}

// ================================================================== 25. thank you
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  s.addText("Thank you", { x: MX, y: 2.2, w: 8, h: 1.2, fontSize: 48, bold: true, color: C.white, fontFace: HEAD, margin: 0, isTextBox: true });
  s.addText("Questions?", { x: MX, y: 3.4, w: 8, h: 0.8, fontSize: 28, color: C.ice, fontFace: HEAD, margin: 0, isTextBox: true });
  s.addText("Code, configs, results and the number-to-file table (docs/presentation/NUMBERS.md) are in the project repository.",
    { x: MX, y: 4.6, w: 8, h: 0.8, fontSize: 16, color: C.white, fontFace: BODY, margin: 0, valign: "top", isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y: 1.05, w: 3.8, h: 4.45, fill: { color: C.white }, line: { color: C.white }, rectRadius: 0.12 });
  image(s, "crop_nz_observed_map.png", { x: 9.1, y: 1.2, w: 3.5, h: 4.15 }, "CORDEX NZ rainfall");
  notes(s, AB, `Thank you. We are happy to take questions. Backup slides follow with the full list of deviations from
    the paper, the hyperparameters, the data provenance, the Weight Network confusion matrices and example maps.`);
}

// ================================================================== backup
{
  const s = content("Backup", "Backup", "Deviations from the paper");
  table(s, [
    ["Item", "Paper", "Ours", "Reason"],
    ["Data (NZ)", "ERA5 → VCSN, 11,491 points", `CORDEX-ML-Bench, ${n("nz.land_cells")} land cells`, "ERA5 / VCSN not available"],
    ["Predictors (NZ)", "T, Q, U, V, W", "T, Q, U, V (16 × 16)", "W850 not in the benchmark"],
    ["Rain threshold", "y > 0", `y > ${n("cfg.nz.rain_threshold")} mm/day (NZ)`, "Numerical drizzle in model output"],
    ["Conv kernel", "6", `${n("nz.kernel")} (NZ); 6, valid (India)`, "16 × 16 too small for 6 × 6"],
    ["Segment bounds", "lᵢ < total < hᵢ", "tᵢ ≤ total < tᵢ₊₁", "Every day in one interval"],
    ["Weight Network loss", "EMD, Eq. 6", "Squared EMD on CDFs", "Eq. 6 collapses"],
    ["India intervals", "Bayesian GMM", "Paper's fixed bounds", `GMM unstable (${n("in.gmm.degenerate")})`],
    ["India season", "All days", "JJAS only", "Data provided for the monsoon"],
    ["Repetitions", n("paper.reps"), n("in.study.reps"), "Compute"],
  ], { x: MX, y: 1.55, w: W - 2 * MX, colW: [2.2, 2.7, 3.6, 3.633], size: 13, rowH: 0.45, name: "deviations table" });
  source(s, "README.md §4 and §8 (Deviations)", 6.25);
  notes(s, AB, `Backup: the full list of deviations from the paper with the reason for each, from the README.`);
}
{
  const s = content("Backup", "Backup", "Hyperparameters");
  table(s, [
    ["Setting", "NZ (configs/cordex_v2.json)", "India (configs/india_v1.json)"],
    ["Channels / kernel", `${n("cfg.nz.channels")} / ${n("cfg.nz.kernel")}`, `${n("cfg.in.channels")} / ${n("in.kernel")} (${n("in.padding")})`],
    ["Members: LR, patience, max epochs", `${n("cfg.nz.member_lr")}, ${n("cfg.nz.patience")}, ${n("cfg.nz.max_epochs")} (up to ${n("cfg.nz.max_epochs_hard")})`, `${n("cfg.in.member_lr")}, ${n("cfg.in.patience")}, ${n("cfg.in.max_epochs")}`],
    ["Warm-start LR", n("cfg.nz.warm_lr"), n("cfg.in.warm_lr")],
    ["LR decay", `${n("cfg.nz.decay")} every ${n("cfg.nz.decay_steps")} steps`, "same"],
    ["Batch size", n("cfg.nz.batch"), n("cfg.in.batch")],
    ["Weight Network: LR, patience, dense, dropout", `${n("cfg.nz.wn_lr")}, ${n("cfg.nz.wn_patience")}, ${n("cfg.nz.wn_dense")}, ${n("cfg.nz.wn_dropout")}`, `${n("cfg.in.wn_lr")}, ${n("cfg.in.wn_patience")}`],
    ["Weight Network loss", n("cfg.nz.wn_loss"), n("cfg.nz.wn_loss")],
    ["Ensemble stage: LR, patience, max epochs", `${n("cfg.nz.ens_lr")}, ${n("cfg.nz.ens_patience")}, ${n("cfg.nz.ens_max_epochs")}`, `${n("cfg.in.ens_lr")}, ${n("cfg.in.ens_patience")}, ${n("cfg.in.ens_max_epochs")}`],
    ["Mass loss", "absolute, initial-ratio scaling", `${n("cfg.in.mass_mode")}, ${n("cfg.in.scaling")}, ε = ${n("in.mass.eps")} mm/day`],
    ["Rain threshold", `${n("cfg.nz.rain_threshold")} mm/day`, `${n("cfg.in.rain_threshold")} (IMD has true zeros)`],
  ], { x: MX, y: 1.55, w: W - 2 * MX, colW: [3.9, 4.1, 4.133], size: 13, rowH: 0.4, name: "hyperparameter table" });
  source(s, `configs/cordex_v2.json, configs/india_v1.json; the NZ member study runs with --sanity (base models ${n("cfg.nz.sanity_member_epochs")} epochs, Weight Network ${n("cfg.nz.sanity_wn_epochs")})`, 6.3);
  notes(s, AB, `Backup: hyperparameters from the two configuration files.`);
}
{
  const s = content("Backup", "Backup", "Data provenance");
  card(s, MX, 1.6, 5.98, 3.3, "India", [
    `ERA5: scripts/download_era5_india_jjas.py (hourly CDS download, daily means, 4 × 4 coarsening), run on two machines; the ${n("in.era5_two_machines")} present on both were byte-identical`,
    "IMD: scripts/slice_imd_jjas.py cuts IMD's 0.25° yearly files to 1 June – 30 September",
    `Checked file by file (scripts/check_india_data.py): ${n("in.files.era5")} + ${n("in.files.imd")} files, ${n("in.cells")}-cell mask in every year`,
  ], { size: 15 });
  card(s, 6.75, 1.6, 5.98, 3.3, "New Zealand", [
    `CORDEX-ML-Bench, Zenodo record ${n("nz.zenodo_v4")} (v4, NZ_domain.zip) and ${n("nz.zenodo_v7")} (v7) for the 1981–2000 target`,
    "Files present in both versions are byte-identical",
    "The v4 1961–1975 data reproduces the mid-term GMM intervals exactly (results/baseline_rerun/data_match_check.json)",
  ], { size: 15 });
  source(s, "README.md §2 and §8 (Provenance); results/india/qre_mass_v1/data_check.json", 5.9);
  notes(s, AB, `Backup: where the data came from and how it was checked.`);
}
{
  const s = content("Backup", "Backup", "Weight Network confusion matrices (validation)");
  image(s, "fig_confusion.png", { x: MX, y: 1.55, w: W - 2 * MX, h: 4.3 }, "Weight Network confusion matrices");
  source(s, "results/fixes_v2/nz_wn/sanity/weight_network_losses/results.json (rows: true interval; columns: predicted)", 6.0);
  notes(s, AB, `Backup: the paper's EMD puts every day in one column; squared EMD gives a near-diagonal matrix with most
    errors one interval away.`);
}
{
  const s = content("Backup", "Backup", "Example test days: observed vs models");
  image(s, "crop_india_maps_row.png", { x: MX, y: 1.5, w: W - 2 * MX, h: 2.3 }, "India example maps");
  image(s, "crop_nz_maps_row.png", { x: MX, y: 3.85, w: W - 2 * MX, h: 2.3 }, "NZ example maps");
  source(s, "results/india/qre_mass_v1/placements/plots/example_maps.png; results/fixes_v2/nz_wn/sanity/mass_study/placements/plots/example_maps.png (first rows)", 6.25);
  notes(s, AB, `Backup: one test day per region, observed rainfall next to BGNet, QRE, the control and QRE plus mass.
    All models smooth the extremes; the mass term changes the totals, not the spatial pattern.`);
}

// ------------------------------------------------------------------ write
const out = path.join(HERE, "BTP_endterm.pptx");
// pptxgenjs writes folder entries into the zip, which some cloud services refuse; rewrite the zip without them
async function dropFolderEntries(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const clean = new JSZip();
  for (const [name, entry] of Object.entries(zip.files)) {
    if (!entry.dir) clean.file(name, await entry.async("uint8array"), { createFolders: false });
  }
  fs.writeFileSync(file, await clean.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

pres.writeFile({ fileName: out }).then(() => dropFolderEntries(out)).then(() => {
  // NUMBERS.md: every number used on a slide or in the notes, with the committed file it comes from
  const esc = (t) => String(t).replace(/\|/g, "\\|").replace(/\n/g, " ");
  const rows = [...USED].sort().map((k) => {
    const e = NUM[k];
    const raw = typeof e.value === "string" ? e.value : JSON.stringify(e.value);
    return `| \`${k}\` | ${esc(e.display)} | ${esc(raw.length > 80 ? raw.slice(0, 77) + "..." : raw)} | \`${esc(e.source)}\` | ${esc(e.field)} |`;
  });
  fs.writeFileSync(path.join(HERE, "NUMBERS.md"), [
    "# Numbers in the end-term presentation and where they come from",
    "",
    "Generated by `build_deck.js` from `figures/numbers.json` (made by `figures/extract_numbers.py` from committed files only).",
    `It lists the ${USED.size} numbers used on the slides and in the speaker notes. "Shown" is the text on the slide; "raw" is the value as`,
    "stored in the source (shown values are rounded only to the precision the committed summaries use: MSE 2 decimals,",
    "MAE and discrepancy 3, p-values 3 significant digits). Text taken from a markdown file is checked to appear verbatim there.",
    "The charts are made from the same `numbers.json` by `figures/make_figures.py`.",
    "",
    "| Key | Shown | Raw | Source file | Field |",
    "| --- | --- | --- | --- | --- |",
    ...rows, "",
  ].join("\n"));
  console.log(`wrote ${out} and NUMBERS.md (${USED.size} numbers used)`);
});
