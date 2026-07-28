from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK = OUTPUT_DIR / "correct_dst_k10_statistical_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "MIL",
    "language": "python",
    "name": "MIL",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Correct DST k=10: patient-paired statistical analysis\n\n"
        "This notebook rebuilds the five-model comparison with the correct GLAM-embedding DST-Prototype k=10. "
        "The primary inference is paired on the same independent test patients; image-level results retain patient clustering."
    ),
    nbf.v4.new_markdown_cell(
        "## Analysis contract\n\n"
        "- Correct DST source: `G:\\\\611\\\\glam\\\\proto_embedding_rerun\\\\DST_k_10`\n"
        "- Primary unit: patient; patient score = mean image probability, patient label = maximum image label.\n"
        "- Primary test: paired DeLong, four prespecified DST-vs-baseline comparisons, Holm correction.\n"
        "- Uncertainty: 20,000 paired stratified bootstraps; robustness: 20,000 paired score-swap permutations.\n"
        "- Image sensitivity analysis: stratified patient-cluster bootstrap and patient-cluster score swap.\n"
        "- Seed: `20260728`; all tests are two-sided with alpha 0.05."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n\n"
        "OUTPUT_DIR = Path.cwd()\n"
        "from analysis import run_analysis\n"
        "results = run_analysis()"
    ),
    nbf.v4.new_markdown_cell("## Forced data and source validation"),
    nbf.v4.new_code_cell(
        "display(pd.DataFrame([results['validation']]).T.rename(columns={0: 'observed'}))\n"
        "display(results['source_audit'])"
    ),
    nbf.v4.new_markdown_cell(
        "The analysis aborts unless the common cohort is exactly 8,409 images, 862 patients, and 19 positive patients; "
        "label conflicts, duplicate keys, missing predictions, and non-finite predictions must all equal zero. "
        "It also aborts unless the correct DST patient AUROC reproduces 0.864831 on all 8,433 images and 0.864581 on the common cohort."
    ),
    nbf.v4.new_markdown_cell("## AUROC, AUPRC, confidence intervals, and descriptive threshold metrics"),
    nbf.v4.new_code_cell(
        "cols = ['grain','model','n','positives','auc','auc_ci95_low','auc_ci95_high','auprc','auprc_ci95_low','auprc_ci95_high',"
        "'sensitivity_0p5','specificity_0p5','bacc_0p5','f1_0p5']\n"
        "display(results['model_metrics'][cols].round(6))"
    ),
    nbf.v4.new_markdown_cell("## Patient-level primary paired inference"),
    nbf.v4.new_code_cell(
        "display(results['patient_tests'].round(6))"
    ),
    nbf.v4.new_markdown_cell(
        "DeLong p values with Holm correction define the primary significance conclusions. "
        "The paired score-swap permutation is a robustness check for the small positive-patient count, not a replacement primary test."
    ),
    nbf.v4.new_markdown_cell("## Image-level patient-cluster sensitivity analysis"),
    nbf.v4.new_code_cell(
        "display(results['image_tests'].round(6))"
    ),
    nbf.v4.new_markdown_cell("## Patient-level ROC curves"),
    nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT_DIR / 'patient_roc.png')))"),
    nbf.v4.new_markdown_cell("## Paired ΔAUROC forest plot"),
    nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT_DIR / 'patient_delta_auc_forest.png')))"),
    nbf.v4.new_markdown_cell("## Paper-ready Statistical Analysis"),
    nbf.v4.new_code_cell("display(Markdown((OUTPUT_DIR / 'statistical_analysis_methods_en.md').read_text(encoding='utf-8')))"),
    nbf.v4.new_markdown_cell("## Paper-ready Results"),
    nbf.v4.new_code_cell("display(Markdown((OUTPUT_DIR / 'results_en.md').read_text(encoding='utf-8')))"),
    nbf.v4.new_markdown_cell("## 中文结论"),
    nbf.v4.new_code_cell("display(Markdown((OUTPUT_DIR / 'summary_zh.md').read_text(encoding='utf-8')))"),
    nbf.v4.new_markdown_cell(
        "## Reproducibility notes\n\n"
        "The companion `analysis.py` contains the full source-loading, common-key construction, DeLong, bootstrap, permutation, Holm correction, plotting, and output code. "
        "Bootstrap sampling is stratified by positive/negative patient; identical resampling weights are used for every model. "
        "No model is retrained, and no result is paired across different training folds."
    ),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
