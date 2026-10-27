"""Train condition-specific probes for main-text C1/C2/C4 endpoints."""
import json
from datetime import datetime
from src.config import ROOT, EXPERIMENTS, CORE_MODELS, LOG_CONFIG, SEEDS
from src.features import run_extraction
from src.probe import run_probe_experiment
from src.statistics import aggregate_patient_predictions, bootstrap_auroc_ci, load_predictions


def run_exp1(models: list = None, datasets: list = None):
    config = EXPERIMENTS["exp1_primary"]
    models = CORE_MODELS if models is None else models
    datasets = config["datasets"] if datasets is None else datasets
    endpoints = {}
    for model in models:
        for dataset in datasets:
            values = {}
            for condition in config["conditions"]:
                run_extraction(model_name=model, dataset_name=dataset, condition=condition)
                summary = run_probe_experiment(model_name=model, dataset_name=dataset, condition=condition, seeds=SEEDS)
                if not summary:
                    raise RuntimeError(f"Probe produced no results: {model}/{dataset}/{condition}")
                predictions = load_predictions(model, dataset, condition)
                if predictions.empty:
                    raise RuntimeError(f"Missing predictions: {model}/{dataset}/{condition}")
                patient = aggregate_patient_predictions(predictions)
                bootstrap = bootstrap_auroc_ci(patient["y_true"].to_numpy(), patient["y_prob"].to_numpy(), patient["patient_id"].to_numpy())
                values[condition] = {
                    "auroc": float(bootstrap["observed_auroc"]),
                    "ci_lower": float(bootstrap["ci_lower"]),
                    "ci_upper": float(bootstrap["ci_upper"]),
                    "n_patients": len(patient),
                }
            endpoints[f"{model}/{dataset}"] = values
    report = {"endpoint": "patient-level pooled OOF AUROC", "settings": endpoints}
    (ROOT / "results" / "exp1_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG_CONFIG["exp1_runlog"].write_text(f"# Primary endpoints\n\nGenerated: {datetime.now().isoformat()}\n\n" + json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_exp1()
