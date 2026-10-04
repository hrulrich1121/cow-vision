/**
 * Builds the results presentation.
 *   node report/make_deck.js
 * Run report/make_figures.py and report/make_model_figures.py first.
 */
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "calf-posture-pipeline.pptx");

// Forest & Moss - agricultural, and it stays out of the way of the charts' blue/orange
const FOREST = "2C5F2D";
const MOSS = "97BC62";
const CREAM = "F5F5F5";
const WHITE = "FFFFFF";
const INK = "111511";
const INK2 = "55605A";
const RULE = "DFE3DA";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Helmut Ulrich";
pres.title = "Calf posture pipeline";

const fig = (f) => path.join(FIG, f);

function contentSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.5, y: 0.28, w: 9, h: 0.24, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 1.6,
    });
  }
  s.addText(title, {
    x: 0.5, y: kicker ? 0.52 : 0.38, w: 9, h: 0.62, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 26, bold: true, color: INK,
  });
  return s;
}

/** Stat card: big number + label. */
function stat(s, x, y, w, value, label, note) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 1.5, rectRadius: 0.08,
    fill: { color: CREAM }, line: { color: RULE, width: 0.75 },
  });
  s.addText(value, {
    x: x + 0.18, y: y + 0.16, w: w - 0.36, h: 0.55, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 30, bold: true, color: FOREST,
  });
  s.addText(label, {
    x: x + 0.18, y: y + 0.74, w: w - 0.36, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, color: INK,
  });
  if (note) {
    s.addText(note, {
      x: x + 0.18, y: y + 1.02, w: w - 0.36, h: 0.38, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10, color: INK2,
    });
  }
}

function bullets(s, x, y, w, items, opts = {}) {
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 },
    })),
    {
      x, y, w, h: opts.h || 2.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: opts.fontSize || 13, color: opts.color || INK,
      lineSpacing: opts.lineSpacing || 19, paraSpaceAfter: 7,
    }
  );
}

function caption(s, text, y) {
  s.addText(text, {
    x: 0.5, y, w: 9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, italic: true, color: INK2,
  });
}

/* ------------------------------------------------------------------ 1 title */
{
  const s = pres.addSlide();
  s.background = { color: FOREST };
  s.addText("Automated lying and standing time from calf pen video", {
    x: 0.7, y: 1.5, w: 8.6, h: 1.5, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, lineSpacing: 42,
  });
  s.addText("Three weeks of CH1 footage, one second at a time", {
    x: 0.7, y: 3.05, w: 8.6, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 16, color: MOSS,
  });
  s.addShape(pres.ShapeType.line, {
    x: 0.72, y: 3.72, w: 1.2, h: 0, line: { color: MOSS, width: 2 },
  });
  s.addText("Helmut Ulrich  ·  Animal Science / AgriLife  ·  September 2026", {
    x: 0.7, y: 3.95, w: 8.6, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, color: CREAM,
  });
  s.addNotes(
    "Four parts: what the pipeline does, how the model was trained, how accurate it is, " +
    "and what the first three weeks of data show."
  );
}

/* ------------------------------------------------------------- 2 what I did */
{
  const s = contentSlide("What the pipeline does", "Method");
  s.addImage({ path: fig("coco_vs_ours.png"), x: 5.35, y: 1.15, w: 4.2, h: 3.91 });
  bullets(s, 0.5, 1.5, 4.6, [
    "Reads the wall-clock time out of every filename, so each frame keeps its real timestamp",
    "Samples 1 frame per second and cuts the left and right pen apart - the two calves never mix",
    "A trained model finds the calf and labels it lying or standing; a 5-frame majority vote removes flicker",
    "Off-the-shelf COCO weights were useless from overhead - the white calf read as a dog, the black one as scissors - so the model was trained on our own labels",
  ], { fontSize: 11.5, h: 3.4, lineSpacing: 17 });
  caption(s, "Same two frames: standard COCO weights above, our trained model below.", 5.15);
  s.addNotes(
    "The whole run: 1.8 M frames sampled, 3.7 M calf crops classified, 502 hours of video " +
    "per pen, unattended on the lab server."
  );
}

/* ------------------------------------------------------------ 3 labelling */
{
  const s = contentSlide("Labelling: 286 by hand, 503 by machine", "Training data");
  stat(s, 0.5, 1.45, 2.85, "789", "images labelled", "4 segments across one day");
  stat(s, 3.55, 1.45, 2.85, "286", "drawn by hand", "in Roboflow, ~2 hours");
  stat(s, 6.6, 1.45, 2.9, "503", "auto-labelled", "16 uncertain ones checked by eye");
  bullets(s, 0.5, 3.2, 9, [
    "Frames sampled every 20 s across pre-dawn, morning, afternoon and night, then near-duplicates dropped - a calf that has not moved in 20 minutes teaches nothing",
    "One box per calf, two classes: lying and standing. Empty pens labelled as \"no calf\"",
    "A first model trained on the 286 hand labels pre-labelled the rest; only the frames it was unsure about came back for human review",
  ], { fontSize: 12, h: 2.0, lineSpacing: 18 });
  s.addNotes(
    "Class balance: 581 lying, 178 standing, 31 empty. Standing is the rarer class, " +
    "which is worth remembering when reading the metrics."
  );
}

/* ------------------------------------------------------------- 4 training */
{
  const s = contentSlide("Training", "The model");
  s.addImage({ path: fig("training_curves.png"), x: 0.5, y: 1.4, w: 6.2, h: 3.1 });
  const specs = [
    ["Architecture", "YOLOv8n (3.2 M parameters)"],
    ["Input", "640 px, letterboxed"],
    ["Split", "704 train / 57 val / 28 test"],
    ["Augmentation", "flips + rotation"],
    ["Hardware", "one RTX 5070, ~2 min"],
  ];
  let y = 1.45;
  for (const [k, v] of specs) {
    s.addText(k, {
      x: 7.0, y, w: 2.5, h: 0.24, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10, bold: true, color: MOSS,
    });
    s.addText(v, {
      x: 7.0, y: y + 0.22, w: 2.5, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, color: INK,
    });
    y += 0.62;
  }
  caption(s, "Validation scores after every epoch. The held-out test split was only touched once, at the end.", 4.7);
  s.addNotes(
    "Validation and test images are human-labelled only - none of the machine-labelled " +
    "frames are in them, so the model is not grading its own homework."
  );
}

/* -------------------------------------------------------------- 5 metrics */
{
  const s = contentSlide("Accuracy on frames the model never saw", "Metrics");
  s.addImage({ path: fig("confusion.png"), x: 0.4, y: 1.35, w: 4.6, h: 3.74 });
  const rows = [
    [{ text: "Class", options: { bold: true } }, { text: "Precision", options: { bold: true } },
     { text: "Recall", options: { bold: true } }, { text: "F1", options: { bold: true } }],
    ["lying", "0.96", "1.00", "0.98"],
    ["standing", "0.96", "1.00", "0.98"],
    ["no calf", "1.00", "0.78", "0.88"],
  ];
  s.addTable(rows, {
    x: 5.3, y: 1.5, w: 4.2, colW: [1.5, 1.0, 0.85, 0.85],
    fontFace: BODY, fontSize: 12, color: INK, border: { type: "solid", color: RULE, pt: 0.75 },
    fill: { color: WHITE }, rowH: 0.34, valign: "middle",
  });
  bullets(s, 5.3, 3.35, 4.2, [
    "Overall 97.6% correct; mAP@50 0.97, mAP@50-95 0.87",
    "Both errors are empty pens where it saw a calf that was not there",
    "Caveat: 85 frames, all from one day and these two calves - the human validation is the real test",
  ], { fontSize: 11.5, h: 1.7, lineSpacing: 17 });
  s.addNotes(
    "Precision = when it says lying, how often it is right. Recall = of the frames that " +
    "really are lying, how many it caught. Not one lying/standing mix-up, but small n."
  );
}

/* --------------------------------------------------------------- 6 trend */
{
  const s = contentSlide("Daily lying time, 19 full days", "Results");
  s.addImage({ path: fig("trend_lying_pct.png"), x: 0.9, y: 1.4, w: 8.2, h: 4.02 });
  s.addNotes(
    "Both calves lie 50-87% of the day and the day-to-day swings are large. The right " +
    "calf drifts down about 0.8 points a day (r = -0.53); the left shows no trend at all. " +
    "With two animals this is a description, not a finding."
  );
}

/* ---------------------------------------------------------------- 7 hours */
{
  const s = contentSlide("The same data as hours per day", "Results");
  s.addImage({ path: fig("daily_hours.png"), x: 1.65, y: 1.35, w: 6.7, h: 4.02 });
  s.addNotes(
    "Left calf averages 16.9 h lying a day, right calf 15.4 h. Bars stop short of 24 h " +
    "where footage is missing."
  );
}

/* -------------------------------------------------------------- 8 diurnal */
if (fs.existsSync(fig("diurnal.png"))) {
  const s = contentSlide("Lying follows the clock", "Results");
  s.addImage({ path: fig("diurnal.png"), x: 0.7, y: 1.45, w: 8.6, h: 3.75 });
  s.addNotes(
    "Pooled over every minute of all 25 dates. The two sharp dips, around 07:00 and " +
    "19:00, are the same on both calves and on every day - the obvious candidate is " +
    "feeding, which the barn schedule can confirm. A classifier that was guessing would " +
    "not produce a repeatable twice-daily pattern, so this doubles as a sanity check."
  );
}

/* --------------------------------------------------------- 9 deliverables */
{
  const s = contentSlide("Files", "Output");
  const files = [
    ["minute_activity.csv", "channel, calf, timestamp (1 min), activity - plus the frame counts behind each minute", "for validation"],
    ["daily_summary_all.csv", "per calf per day: lying / standing hours, observed hours, detection rate", "the charts"],
    ["bouts_all.csv", "every continuous lying or standing stretch over a minute", "welfare metrics"],
  ];
  let y = 1.6;
  for (const [name, desc, tag] of files) {
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.5, y, w: 9, h: 0.9, rectRadius: 0.06,
      fill: { color: CREAM }, line: { color: RULE, width: 0.75 },
    });
    s.addText(name, {
      x: 0.75, y: y + 0.14, w: 3.3, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 12, bold: true, color: FOREST,
    });
    s.addText(desc, {
      x: 0.75, y: y + 0.47, w: 6.6, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, color: INK2,
    });
    s.addText(tag, {
      x: 7.6, y: y + 0.31, w: 1.65, h: 0.28, isTextBox: true, margin: 0, align: "right",
      fontFace: BODY, fontSize: 10.5, bold: true, color: MOSS,
    });
    y += 1.05;
  }
  s.addText("61,146 minutes across 25 dates, both calves.", {
    x: 0.5, y: 4.9, w: 9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, color: INK,
  });
  s.addNotes(
    "Each minute in minute_activity.csv also carries how many of its 60 frames agreed, " +
    "so a disputed minute can be judged on the strength of the evidence. 7.6% of minutes " +
    "fall below 80% agreement - those are the transitions, and the fairest place for a " +
    "human to disagree."
  );
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
