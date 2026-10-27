"""Train original-mask morphology controls reported in the main text."""
import json
from datetime import datetime
from src.config import ROOT, EXPERIMENTS, LOG_CONFIG, SEEDS
from src.features import run_extraction
from src.probe import run_probe_experiment
from src.statistics import aggregate_patient_predictions, align_patient_predictions, load_predictions, paired_bootstrap_difference


def run_exp2(models: list = None, datasets: list = None, conditions: list = None):
    config = EXPERIMENTS["exp2_controls"]
    models = config["models"] if models is None else models
    datasets = config["datasets"] if datasets is None else datasets
    conditions = config["conditions"] if conditions is None else conditions
    if not conditions or conditions[0] != "C4_lesion-removed-inpaint" or set(conditions) - set(config["conditions"]):
        raise ValueError("Morphology controls require C4 first and the registered original-mask conditions")
    reference = conditions[0]
    comparisons = {}
    for model in models:
        for dataset in datasets:
            predictions = {}
            for condition in conditions:
                run_extraction(model_name=model, dataset_name=dataset, condition=condition)
                summary = run_probe_experiment(model_name=model, dataset_name=dataset, condition=condition, seeds=SEEDS)
                if not summary:
                    raise RuntimeError(f"Probe produced no results: {model}/{dataset}/{condition}")
                frame = load_predictions(model, dataset, condition)
                if frame.empty:
                    raise RuntimeError(f"Missing predictions: {model}/{dataset}/{condition}")
                predictions[condition] = aggregate_patient_predictions(frame)
            setting = {}
            for condition in conditions[1:]:
                aligned = align_patient_predictions(predictions[condition], predictions[reference])
                if aligned.empty:
                    raise RuntimeError(f"No aligned patients: {condition} and {reference}")
                delta = paired_bootstrap_difference(
                    y_true=aligned["y_true"].to_numpy(), y_prob_1=aligned["y_prob_1"].to_numpy(),
                    y_prob_2=aligned["y_prob_2"].to_numpy(), patient_ids=aligned["patient_id"].to_numpy(),
                )
                setting[condition] = {key: float(delta[key]) for key in ("delta_auroc", "ci_lower", "ci_upper")}
            comparisons[f"{model}/{dataset}"] = setting
    report = {"contrast": "control minus C4", "interval": "nominal paired patient bootstrap 95% CI", "settings": comparisons}
    (ROOT / "results" / "exp2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG_CONFIG["exp2_runlog"].write_text(f"# Original-mask morphology controls\n\nGenerated: {datetime.now().isoformat()}\n\n" + json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_exp2()
