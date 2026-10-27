"""Run main-text infrastructure, primary endpoints, or morphology controls."""
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"


def _stage_log_path(name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.lower()).strip("_")
    return LOGS_DIR / f"{safe_name}_pipeline.log"


def run_stage(name: str, stage_fn):
    """Persist status; the CLI propagates failure as a nonzero exit."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = stage_fn()
        status = "OK"
    except Exception as exc:
        result = {"error": str(exc)}
        status = "FAILED"
        traceback.print_exc()
    _stage_log_path(name).write_text(
        f"Stage: {name}\nStatus: {status}\nTime: {datetime.now().isoformat()}\nResult: {result}\n",
        encoding="utf-8",
    )
    print(f"[{status}] {name}")
    return {"status": status, "result": result}


def stage_infrastructure():
    from src import conditions, manifest, masks, splits
    manifest.main()
    masks.process_all_masks()
    generated = conditions.main()
    if generated is None or generated.empty:
        raise RuntimeError("No condition images were generated")
    for condition in conditions.CONDITIONS:
        if condition not in generated or generated[condition].isna().any() or generated[condition].eq("").any():
            raise RuntimeError(f"Condition generation incomplete: {condition}")
    splits.main()
    return {"images": len(generated), "conditions": list(conditions.CONDITIONS)}


def stage_exp1(models=None, datasets=None):
    from src.exp1_primary import run_exp1
    return run_exp1(models=models, datasets=datasets)


def stage_exp2(models=None, datasets=None):
    from src.exp2_controls import run_exp2
    return run_exp2(models=models, datasets=datasets)


STAGE_MAP = {
    "infra": ("Infrastructure", stage_infrastructure),
    "exp1": ("Primary endpoints", stage_exp1),
    "exp2": ("Original-mask morphology controls", stage_exp2),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGE_MAP))
    parser.add_argument("--models", nargs="+", choices=["ResNet-50", "DINOv3", "MedSAM2", "UltraSam"])
    parser.add_argument("--datasets", nargs="+", choices=["BUS-BRA", "BUS-UCLM"])
    args = parser.parse_args(argv)
    if args.stage == "infra" and (args.models or args.datasets):
        parser.error("infra processes installed BUS-BRA/BUS-UCLM data; filters apply to exp1 or exp2")
    name, stage_fn = STAGE_MAP[args.stage]
    if args.stage != "infra":
        stage_fn = partial(stage_fn, models=args.models, datasets=args.datasets)
    return 0 if run_stage(name, stage_fn)["status"] == "OK" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
