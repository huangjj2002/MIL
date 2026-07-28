from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from edl_proto_train import (
    _prototype_diversity_loss,
    _resolve_proto_margins,
    _select_fold_best_score_rows,
    _single_output_proto_reg,
)
from analyze_embedding_dst_ablation import binary_auc_auprc
from run_embedding_dst_ablation import (
    REVIEWER17,
    REVIEWER9,
    build_command,
    choose_variants,
    validate_cache,
)


class EmbeddingDSTAblationTests(unittest.TestCase):
    def test_reviewer9_definition_and_cache_preflight(self) -> None:
        self.assertEqual(len(REVIEWER9), 9)
        self.assertEqual(len(REVIEWER17), 17)
        self.assertEqual(len({variant.name for variant in REVIEWER17}), 17)
        self.assertEqual(choose_variants("full_k10,k5")[1].k, 5)
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / "manifest.json").write_text(
                json.dumps({"embedding_level": "bag_origin"}), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "patient_id": ["p0", "p1"],
                    "image_id": ["i0", "i1"],
                    "embedding_start": [0, 1],
                    "embedding_end": [1, 2],
                    "origin_prediction_score": [0.1, 0.9],
                }
            ).to_csv(cache_dir / "metadata.csv", index=False)
            np.save(cache_dir / "embeddings.npy", np.ones((2, 4), dtype=np.float32))
            result = validate_cache(cache_dir)
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["embedding_shape"], [2, 4])
            self.assertEqual(len(result["metadata_sha256"]), 64)

        command_args = Namespace(
            python=sys.executable,
            trainer=REPO_ROOT / "edl_proto_train.py",
            gpu_id="0",
            data_dir="data",
            csv_file="data.csv",
            embedding_cache_dir=Path("cache"),
            dataset="ViNDr",
            label="cancer",
            n_folds=5,
            train_cohorts="1-8",
            test_cohorts="9-10",
            seed=10,
            epochs=1,
            batch_size=8,
            lr=3e-4,
            weight_decay=1e-4,
            num_workers=0,
            extra_trainer_args=[],
        )
        command = build_command(command_args, REVIEWER9[0], Path("run"))
        joined = " ".join(command)
        self.assertIn("--edl_proto_init fold_best_scores", joined)
        self.assertIn("--edl_proto_gamma_sep 1.0", joined)
        self.assertIn("--edl_proto_gamma_div 1.0", joined)

    def test_separate_margins_and_legacy_fallback(self) -> None:
        legacy = Namespace(
            edl_proto_margin=0.7,
            edl_proto_gamma_sep=None,
            edl_proto_gamma_div=None,
        )
        self.assertEqual(_resolve_proto_margins(legacy), (0.7, 0.7))
        override = Namespace(
            edl_proto_margin=0.7,
            edl_proto_gamma_sep=0.2,
            edl_proto_gamma_div=1.2,
        )
        self.assertEqual(_resolve_proto_margins(override), (0.2, 1.2))

        import torch

        labels = torch.tensor([0, 1])
        distances = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.5, 0.9], [0.1, 0.3]],
            ],
            dtype=torch.float32,
        )
        edl_out = {"prototype_distances": distances, "prob": torch.ones(2, 2) / 2}
        _, separation_small, _ = _single_output_proto_reg(edl_out, labels, 0.5)
        _, separation_large, _ = _single_output_proto_reg(edl_out, labels, 1.0)
        self.assertGreater(float(separation_large), float(separation_small))

        class Head:
            normalize = False
            prototypes = torch.tensor([[[0.0, 0.0], [0.5, 0.0]]])

        class Model:
            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def prototype_heads(self):
                return {"head": Head()}

        diversity_small = _prototype_diversity_loss(Model(), 0.2)
        diversity_large = _prototype_diversity_loss(Model(), 1.0)
        self.assertEqual(float(diversity_small), 0.0)
        self.assertGreater(float(diversity_large), float(diversity_small))

        labels_np = np.array([0, 1, 0, 1, 1, 0])
        scores_np = np.array([0.1, 0.5, 0.5, 0.7, 0.9, 0.2])
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc, auprc = binary_auc_auprc(labels_np, scores_np)
        self.assertAlmostEqual(auc, roc_auc_score(labels_np, scores_np), places=12)
        self.assertAlmostEqual(
            auprc, average_precision_score(labels_np, scores_np), places=12
        )

    def test_fold_best_scores_order_and_patient_isolation(self) -> None:
        metadata = pd.DataFrame(
            {
                "patient_id": ["n_good", "n_wrong", "p_low", "p_high", "v", "t"],
                "image_id": ["i0", "i1", "i2", "i3", "iv", "it"],
                "embedding_start": range(6),
                "embedding_end": range(1, 7),
                "origin_prediction_score": [0.1, 0.8, 0.7, 0.9, 0.4, 0.6],
                "origin_predicted_class": [0, 1, 1, 1, 0, 1],
            }
        )
        train = pd.DataFrame(
            {
                "patient_id": ["n_good", "n_wrong", "p_low", "p_high"],
                "image_id": ["i0", "i1", "i2", "i3"],
                "cancer": [0, 0, 1, 1],
            }
        )
        val = pd.DataFrame({"patient_id": ["v"], "image_id": ["iv"], "cancer": [0]})
        test = pd.DataFrame({"patient_id": ["t"], "image_id": ["it"], "cancer": [1]})
        selected = _select_fold_best_score_rows(metadata, train, val, test, "cancer", 2)
        class_zero = selected[selected["prototype_class"] == 0]
        class_one = selected[selected["prototype_class"] == 1]
        self.assertEqual(class_zero.iloc[0]["patient_id"], "n_good")
        self.assertEqual(class_one.iloc[0]["patient_id"], "p_high")
        self.assertEqual(set(selected["patient_id"]), set(train["patient_id"]))
        overlapping_val = val.copy()
        overlapping_val.loc[0, "patient_id"] = "n_good"
        with self.assertRaisesRegex(ValueError, "overlap"):
            _select_fold_best_score_rows(
                metadata, train, overlapping_val, test, "cancer", 1
            )

    def test_reviewer9_dry_run_manifest_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            (cache / "manifest.json").write_text(
                json.dumps({"embedding_level": "bag_origin"}), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "patient_id": ["p0", "p1"],
                    "image_id": ["i0", "i1"],
                    "embedding_start": [0, 1],
                    "embedding_end": [1, 2],
                    "origin_prediction_score": [0.1, 0.9],
                }
            ).to_csv(cache / "metadata.csv", index=False)
            np.save(cache / "embeddings.npy", np.ones((2, 4), dtype=np.float32))
            run_root = root / "run"
            base_command = [
                sys.executable,
                str(REPO_ROOT / "run_embedding_dst_ablation.py"),
                "--embedding-cache-dir",
                str(cache),
                "--run-root",
                str(run_root),
                "--dry-run",
            ]
            first = subprocess.run(base_command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["preset"], "reviewer9")
            self.assertEqual(len(manifest["variants"]), 9)
            self.assertEqual(manifest["fixed_settings"]["prototype_init"], "fold_best_scores")
            resumed = subprocess.run(
                [*base_command, "--resume"], capture_output=True, text=True, check=False
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)

            fake_trainer = root / "fake_trainer.py"
            fake_trainer.write_text("print('fake training completed')\n", encoding="utf-8")
            resume_root = root / "resume_run"
            execution_command = [
                sys.executable,
                str(REPO_ROOT / "run_embedding_dst_ablation.py"),
                "--embedding-cache-dir",
                str(cache),
                "--run-root",
                str(resume_root),
                "--trainer",
                str(fake_trainer),
                "--variants",
                "full_k10",
            ]
            executed = subprocess.run(
                execution_command, capture_output=True, text=True, check=False
            )
            self.assertEqual(executed.returncode, 0, msg=executed.stdout + executed.stderr)
            status = json.loads((resume_root / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["full_k10"]["state"], "completed")
            self.assertEqual(status["full_k10"]["gpu_id"], "2")
            skipped = subprocess.run(
                [*execution_command, "--resume"], capture_output=True, text=True, check=False
            )
            self.assertEqual(skipped.returncode, 0, msg=skipped.stdout + skipped.stderr)
            self.assertIn("already completed; skipping", skipped.stdout)

            parallel_root = root / "parallel_run"
            parallel = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_embedding_dst_ablation.py"),
                    "--embedding-cache-dir",
                    str(cache),
                    "--run-root",
                    str(parallel_root),
                    "--trainer",
                    str(fake_trainer),
                    "--variants",
                    "full_k10,k5",
                    "--gpu-ids",
                    "3",
                    "4",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(parallel.returncode, 0, msg=parallel.stdout + parallel.stderr)
            parallel_status = json.loads(
                (parallel_root / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {parallel_status["full_k10"]["gpu_id"], parallel_status["k5"]["gpu_id"]},
                {"3", "4"},
            )

    def test_analysis_end_to_end_with_synthetic_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            for variant_index, variant in enumerate(REVIEWER9):
                result_dir = (
                    run_root
                    / variant.name
                    / "DST_PROTO"
                    / "ViNDr_cancer"
                    / "fold_5"
                    / "2026-01-01"
                    / "dst_proto_test_results"
                )
                result_dir.mkdir(parents=True)
                validation_rows = []
                test_rows = []
                rng = np.random.default_rng(100 + variant_index)
                separation = 0.30 - 0.018 * variant_index
                for fold in range(5):
                    for split, rows, patient_prefix, patient_count in (
                        ("val", validation_rows, f"v{fold}", 16),
                        ("test", test_rows, "t", 20),
                    ):
                        for patient_index in range(patient_count):
                            label = int(patient_index >= patient_count // 2)
                            patient_id = f"{patient_prefix}_p{patient_index}"
                            patient_score = np.clip(
                                0.5 + (1 if label else -1) * separation + rng.normal(0, 0.16),
                                0.001,
                                0.999,
                            )
                            for image_index in range(2):
                                rows.append(
                                    {
                                        "patient_id": patient_id,
                                        "image_id": f"{patient_id}_i{image_index}",
                                        "split": split,
                                        "cancer": label,
                                        "prediction_score": float(
                                            np.clip(patient_score + (image_index - 0.5) * 0.02, 0.001, 0.999)
                                        ),
                                        "fold": fold,
                                    }
                                )
                pd.DataFrame(validation_rows).to_csv(
                    result_dir / "ViNDr_dst_proto_dev_predictions.csv", index=False
                )
                pd.DataFrame(test_rows).to_csv(
                    result_dir / "ViNDr_dst_proto_test_all_folds.csv", index=False
                )

            command = [
                sys.executable,
                str(REPO_ROOT / "analyze_embedding_dst_ablation.py"),
                "--run-root",
                str(run_root),
                "--bootstrap-samples",
                "100",
                "--permutation-samples",
                "100",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            analysis_dir = run_root / "analysis"
            expected = {
                "fold_metrics.csv",
                "fold_metrics_summary.csv",
                "ensemble_metrics.csv",
                "validation_thresholds.csv",
                "paired_wilcoxon_vs_full.csv",
                "paired_bootstrap_vs_full.csv",
                "patient_primary_tests.csv",
                "parameter_sensitivity_metrics.csv",
                "component_ablation_metrics.csv",
                "ablation_results.md",
                "analysis_manifest.json",
                "ablation_auroc_bacc.png",
                "ablation_auroc_delta_vs_full.png",
            }
            self.assertTrue(expected.issubset({path.name for path in analysis_dir.iterdir()}))
            import matplotlib.image as mpimg

            for plot_name in ("ablation_auroc_bacc.png", "ablation_auroc_delta_vs_full.png"):
                image = mpimg.imread(analysis_dir / plot_name)
                self.assertGreaterEqual(image.shape[0], 800)
                self.assertGreaterEqual(image.shape[1], 1200)
                self.assertGreater(float(np.std(image)), 0.01)
            fold_metrics = pd.read_csv(analysis_dir / "fold_metrics.csv")
            self.assertEqual(len(fold_metrics), 45)
            self.assertEqual(set(fold_metrics["threshold_source"]), {"validation_fold"})
            ensemble = pd.read_csv(analysis_dir / "ensemble_metrics.csv")
            self.assertEqual(len(ensemble), 9)
            self.assertEqual(set(ensemble["threshold_source"]), {"pooled_oof_validation"})
            self.assertIn("auprc", ensemble.columns)
            primary = pd.read_csv(analysis_dir / "patient_primary_tests.csv")
            self.assertIn("p_delong_holm_within_family", primary.columns)
            self.assertIn("p_permutation_auroc_holm_within_family", primary.columns)
            no_attraction_families = set(
                primary.loc[primary["variant"] == "no_attraction", "family"]
            )
            self.assertEqual(
                no_attraction_families,
                {"component", "lambda_attraction"},
            )
            wilcoxon = pd.read_csv(analysis_dir / "paired_wilcoxon_vs_full.csv")
            self.assertEqual(
                set(wilcoxon["minimum_two_sided_exact_p_if_all_five_nonzero"]),
                {0.0625},
            )
            analysis_manifest = json.loads(
                (analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(analysis_manifest["bootstrap_samples"], 100)
            self.assertEqual(analysis_manifest["permutation_samples"], 100)


if __name__ == "__main__":
    unittest.main()
