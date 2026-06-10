import fs from "node:fs";
import path from "node:path";
import { Presentation, PresentationFile, fr } from "file:///C:/Users/hjj/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const inputCsv = "G:/610/analysis/test_only_metrics_recomputed_20260610.csv";
const finalPptx = "G:/610/analysis/g610_test_ensemble_metrics_table_20260610.pptx";
const previewPng = "G:/Final_MIL/code/outputs/019eb08b-5e5e-7830-9fa8-ef74f73f9497/presentations/g610-metrics-table/preview/g610_test_ensemble_metrics_table.png";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && inQuotes && next === '"') {
      cell += '"';
      i++;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i++;
      row.push(cell);
      if (row.some((v) => v.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const header = rows.shift();
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function num(row, key) {
  return Number(row[key]);
}

function f3(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "";
}

function ci3(text) {
  const m = /\[([0-9.]+),\s*([0-9.]+)\]/.exec(text ?? "");
  return m ? `[${f3(Number(m[1]))}, ${f3(Number(m[2]))}]` : "";
}

function metricCell(row, metric) {
  const point = f3(num(row, metric));
  const mean = f3(num(row, `${metric}_fold_mean`));
  const std = f3(num(row, `${metric}_fold_std`));
  const ci = ci3(row[`${metric}_ci95`]);
  return `${point}\nmean ${mean} ± ${std}\n95%CI ${ci}`;
}

const data = parseCsv(fs.readFileSync(inputCsv, "utf8"));
const labelMap = new Map([
  ["DST_PROTO_k10_2026-06-09", "DST k=10\n2026-06-09"],
  ["DST_PROTO_k5_2026-06-10", "DST k=5\n2026-06-10"],
  ["EDL_PROTO_DIRICHLET_k10_2026-06-09", "EDL k=10\n2026-06-09"],
  ["EDL_PROTO_DIRICHLET_k5_2026-06-10", "EDL k=5\n2026-06-10"],
]);

const values = [
  ["Run", "AUC", "bACC", "Sensitivity", "Specificity"],
  ...data.map((row) => [
    labelMap.get(row.run_label) ?? row.run_label,
    metricCell(row, "auc"),
    metricCell(row, "bacc"),
    metricCell(row, "sensitivity"),
    metricCell(row, "specificity"),
  ]),
];

const presentation = Presentation.create();
const slide = presentation.slides.add();
slide.setViewportSize(1280, 720);
slide.background.fill = { type: "solid", color: "#F7F8FA" };

const title = slide.shapes.add({
  geometry: "rect",
  position: { left: 48, top: 34, width: 900, height: 52 },
  fill: { type: "solid", color: "#F7F8FA" },
  line: { width: 0, fill: "#F7F8FA" },
});
title.text = "G:\\610 Test Ensemble Metrics";
title.text.fontSize = 30;
title.text.bold = true;
title.text.typeface = "Aptos Display";
title.text.color = "#172033";

const subtitle = slide.shapes.add({
  geometry: "rect",
  position: { left: 50, top: 84, width: 1120, height: 34 },
  fill: { type: "solid", color: "#F7F8FA" },
  line: { width: 0, fill: "#F7F8FA" },
});
subtitle.text = "Image-level test set only. Each metric cell shows ensemble point estimate, 5-fold test mean ± std, and stratified bootstrap 95%CI.";
subtitle.text.fontSize = 13;
subtitle.text.typeface = "Aptos";
subtitle.text.color = "#596579";

const table = slide.tables.add({
  rows: values.length,
  columns: values[0].length,
  left: 48,
  top: 136,
  width: 1184,
  height: 458,
  columnTracks: [fr(1.2), fr(1), fr(1), fr(1), fr(1)],
  values,
});
table.styleOptions = { headerRow: true, bandedRows: true, firstColumn: true };
table.cellMargins = { top: 7, right: 8, bottom: 7, left: 8 };

const all = table.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: values[0].length });
all.assign({
  textStyle: {
    typeface: "Aptos",
    fontSize: 12,
    color: "#172033",
    alignment: "center",
    verticalAlignment: "middle",
  },
  borders: {
    top: { width: 0.6, color: "#D7DDE7" },
    bottom: { width: 0.6, color: "#D7DDE7" },
    left: { width: 0.6, color: "#D7DDE7" },
    right: { width: 0.6, color: "#D7DDE7" },
  },
});

const header = table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length });
header.assign({
  fill: "#1E3A5F",
  textStyle: { bold: true, color: "#FFFFFF", fontSize: 13, typeface: "Aptos" },
});

const firstCol = table.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: 1 });
firstCol.assign({
  fill: "#EAF0F7",
  textStyle: { bold: true, color: "#172033", fontSize: 12, typeface: "Aptos" },
});

for (let r = 1; r < values.length; r++) {
  const fill = r % 2 === 1 ? "#FFFFFF" : "#F3F6FA";
  table.cells.block({ row: r, column: 1, rowCount: 1, columnCount: values[0].length - 1 }).assign({
    fill,
    textStyle: { fontSize: 11.2, typeface: "Aptos", color: "#172033", alignment: "center", verticalAlignment: "middle" },
  });
}

const note = slide.shapes.add({
  geometry: "rect",
  position: { left: 50, top: 612, width: 1110, height: 42 },
  fill: { type: "solid", color: "#F7F8FA" },
  line: { width: 0, fill: "#F7F8FA" },
});
note.text = "Source: G:\\610\\analysis\\test_only_metrics_recomputed_20260610.csv. Threshold = 0.5; bootstrap seed = 20260610; n = 8,409 images (66 positive / 8,343 negative).";
note.text.fontSize = 11;
note.text.typeface = "Aptos";
note.text.color = "#596579";

const blob = await PresentationFile.exportPptx(presentation);
await blob.save(finalPptx);

const png = await presentation.export({ format: "png", scale: 1 });
fs.writeFileSync(previewPng, Buffer.from(await png.arrayBuffer()));

console.log(finalPptx);
console.log(previewPng);
