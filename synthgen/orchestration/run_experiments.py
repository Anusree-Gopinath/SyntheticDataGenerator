"""Run the full experiment plan with one command.

Run: ``python -m synthgen.orchestration.run_experiments --plan config/experiment_plan.yaml``
(or ``make experiments``).

For each scenario in the plan, every approach runs once per seed through the same
orchestrator the app uses (every row checked against the approved rules, failing rows
regenerated), then each run is evaluated. Each scenario becomes one batch under
``data/runs/`` in the same layout the app writes, so the Report screen can open it too, plus
an ``experiment_report/`` folder with:

- ``runs.csv``: every metric for every run;
- ``summary.csv``: mean and 95% confidence interval per approach and metric;
- ``significance.csv``: paired t-tests between approaches across seeds, Holm-adjusted;
- ``report.html``: the three tables on one page.

Options override the plan for quick checks (for example ``--rows 500 --seeds 1 2``); the
plan file itself is never changed. Only approved profiles are used. An approach that fails
for one seed does not stop the others; an LLM approach is skipped, with the reason, when the
local Ollama server or its model is not available.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from synthgen.core.models import SchemaProfile
from synthgen.core.profile_manager import coerce_sample
from synthgen.evaluation.evaluate import EvaluationStores, evaluate_run, reference_baseline
from synthgen.evaluation.report import build_report, compare_approaches, write_report
from synthgen.gateway import LLMGateway
from synthgen.orchestration.approaches import APPROACHES, Approach
from synthgen.orchestration.orchestrator import run_generator
from synthgen.reference_world.stores import calibration_for_profile, sample_for_profile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROFILE_STORE = DATA / "profiles"
# Plan names that differ from the app's approach keys
PLAN_NAMES = {"llm": "llm_small"}
# Metrics tested for differences between approaches (privacy ones only exist in Scenario 2)
TESTED = [
    "rows_rejected",
    "held_back_gap",
    "reference_distance",
    "tstr_accuracy",
    "seconds",
    "near_copy_share",
    "dcr_median",
    "distance_attack_auc",
    "density_attack_auc",
]


@dataclass
class PlannedApproach:
    approach: Approach
    models: dict[str, str]  # model tags by size, as in config/models.yaml
    scenarios: set[int]


def resolve_profile(spec: str) -> tuple[SchemaProfile, Path]:
    """A profile from a YAML path, ``id@version``, or ``id`` (latest saved version)."""
    path = Path(spec)
    if not path.is_absolute():
        path = ROOT / path
    if spec.endswith((".yaml", ".yml")):
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {spec}")
    elif "@" in spec:
        profile_id, version = spec.rsplit("@", 1)
        path = PROFILE_STORE / profile_id / f"v{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Profile {spec} is not in {PROFILE_STORE}")
    else:
        versions = [p for p in (PROFILE_STORE / spec).glob("v*.yaml") if p.stem[1:].isdigit()]
        if not versions:
            raise FileNotFoundError(f"No saved versions of profile '{spec}' in {PROFILE_STORE}")
        path = max(versions, key=lambda p: int(p.stem[1:]))
    return SchemaProfile.load(path), path


def plan_approaches(plan: dict, models: dict) -> list[PlannedApproach]:
    """The plan's approaches matched to the app's, with the plan's model tags."""
    by_key = {a.key: a for a in APPROACHES}
    planned = []
    for entry in plan["approaches"]:
        key = PLAN_NAMES.get(entry["name"], entry["name"])
        if key not in by_key:
            raise ValueError(f"Unknown approach in the plan: '{entry['name']}'")
        approach = by_key[key]
        tags = dict(models.get("models", {}))
        if approach.model_size and entry.get("model"):
            tags[approach.model_size] = entry["model"]
        scenarios = set(entry.get("scenarios", approach.scenarios)) & set(approach.scenarios)
        planned.append(PlannedApproach(approach, tags, scenarios))
    return planned


def load_stores(scenario: int) -> EvaluationStores:
    """What the evaluators may read. Never passed to generators."""
    thresholds = yaml.safe_load((ROOT / "config" / "thresholds.yaml").read_text())
    stores = EvaluationStores(
        reference=pd.read_csv(DATA / "reference" / "reference.csv", dtype=str),
        held_back=yaml.safe_load((DATA / "evaluation" / "held_back_stats.yaml").read_text()),
        privacy_percentile=float(thresholds["privacy_distance"]["percentile"]),
    )
    if scenario == 2:
        stores.sample = pd.read_csv(DATA / "calibration" / "scenario2_sample.csv", dtype=str)
        stores.candidates = pd.read_csv(DATA / "evaluation" / "candidate_pool.csv", dtype=str)
    return stores


def unavailable(p: PlannedApproach, gateway: LLMGateway, pulled: set[str] | None) -> str | None:
    """Why an approach can't run here, or None."""
    if p.approach.model_size:
        tag = p.models.get(p.approach.model_size)
        if not pulled:
            return "the local Ollama server is not running (start it with `ollama serve`)"
        if tag not in pulled:
            return f"{tag} is not pulled into Ollama (`ollama pull {tag}`)"
    return None


def run_scenario(
    scenario: int,
    planned: list[PlannedApproach],
    profile: SchemaProfile,
    n_rows: int,
    seeds: list[int],
    plan_name: str,
    gateway: LLMGateway,
    out_root: Path,
) -> Path:
    """Generate, save and evaluate every run for one scenario. Returns the batch folder."""
    calibration = calibration_for_profile(
        yaml.safe_load((DATA / "calibration" / "calibration_stats.yaml").read_text()), profile
    )
    sample = None
    if scenario == 2:
        sample = sample_for_profile(
            pd.read_csv(DATA / "calibration" / "scenario2_sample.csv", dtype=str), profile
        )

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    batch = f"{stamp}-s{scenario}-{plan_name}"
    folder = out_root / batch
    folder.mkdir(parents=True, exist_ok=False)
    profile.save(folder / "profile.yaml")  # the exact profile these runs used

    pulled = gateway.available_models() if any(p.approach.model_size for p in planned) else None
    chosen, skipped = [], {}
    for p in planned:
        if scenario not in p.scenarios:
            continue
        reason = unavailable(p, gateway, pulled)
        if reason:
            skipped[p.approach.key] = reason
            print(f"  skip {p.approach.label}: {reason}")
        else:
            chosen.append(p)

    stores = load_stores(scenario)
    _, tstr_test, reference_accuracy = reference_baseline(load_stores(1), profile)
    records, rows, summary_by_approach = [], [], {}
    for seed in seeds:
        for p in chosen:
            a = p.approach
            state = summary_by_approach.setdefault(
                a.key,
                {"label": a.label, "status": "Done", "done": 0, "valid": 0, "rejected": 0,
                 "seconds": 0.0, "incomplete": False},
            )
            if state["status"] == "Failed":
                continue
            try:
                generator = a.make(gateway, p.models, seed)
                if hasattr(generator, "on_batch"):  # LLM runs take hours: show progress
                    generator.on_batch = lambda have, want, label=a.label, seed=seed: print(
                        f"  {label:<22} seed {seed}: {have:,} of {want:,} rows from the model",
                        end="\r",
                        flush=True,
                    )
                data, record = run_generator(
                    generator, profile, calibration, scenario, n_rows, seed, sample
                )
            # Same as the app: one failed approach must not stop the others
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                state["status"] = "Failed"
                print(f"  {a.label:<22} seed {seed}: FAILED ({exc})")
                continue
            record.llm_model = getattr(generator, "model", None)
            csv = folder / f"{record.run_id}.csv"
            data.to_csv(csv, index=False)
            (folder / f"{record.run_id}.json").write_text(record.to_json())
            if getattr(generator, "loss_history", None):  # CTGAN or TVAE training loss per epoch
                pd.DataFrame(generator.loss_history).to_csv(
                    folder / f"{record.run_id}.loss.csv", index=False
                )
            records.append(asdict(record))
            state["done"] += 1
            state["valid"] = len(data)
            state["rejected"] += record.rows_rejected
            state["seconds"] += record.duration_seconds or 0.0
            state["incomplete"] |= not record.complete
            if state["incomplete"]:
                state["status"] = "Incomplete"

            # Evaluate the saved file exactly as the Report screen does
            saved = pd.read_csv(csv, dtype=str, keep_default_na=False, na_values=[""])
            metrics = evaluate_run(coerce_sample(saved, profile.columns), profile, stores, tstr_test)
            rows.append(
                {
                    "run_id": record.run_id,
                    "approach": a.label,
                    "seed": seed,
                    "rows_rejected": record.rows_rejected,
                    "seconds": record.duration_seconds,
                }
                | metrics
            )
            print(
                f"  {a.label:<22} seed {seed}: {len(data):,} rows, "
                f"{record.rows_rejected:,} rejected, {record.duration_seconds:.1f} s"
                + ("" if record.complete else "  (incomplete)")
            )

    manifest = {
        "batch": batch,
        "folder": str(folder),
        "profile": f"{profile.profile_id}@{profile.version}",
        "scenario": scenario,
        "n_rows": n_rows,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "plan": plan_name,
        "skipped": skipped,
        "records": records,
        "approaches": summary_by_approach,
    }
    (folder / "batch.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    if rows:
        runs = pd.DataFrame(rows)
        summary = build_report(runs)
        significance = compare_approaches(runs, TESTED)
        baseline = pd.DataFrame(
            [{"note": "Train on the reference world, test on the same held-out half",
              "tstr_accuracy": reference_accuracy}]
        )
        html = write_report(
            {"summary": summary, "significance": significance, "runs": runs,
             "reference_baseline": baseline},
            folder,
            f"SynthGen experiment: {plan_name}, Scenario {scenario}",
            subfolder="experiment_report",
        )
        print_summary(summary, significance)
        print(f"  Report: {html.relative_to(ROOT) if html.is_relative_to(ROOT) else html}")
    return folder


def print_summary(summary: pd.DataFrame, significance: pd.DataFrame) -> None:
    headline = ["rule_compliance", "rows_rejected", "held_back_gap", "reference_distance",
                "tstr_accuracy", "near_copy_share", "distance_attack_auc", "seconds"]
    table = (
        summary[summary["metric"].isin(headline)]
        .pivot(index="metric", columns="approach", values="mean")
        .reindex([m for m in headline if m in set(summary["metric"])])
    )
    print("\n  Mean across seeds")
    print("  " + table.round(3).to_string().replace("\n", "\n  "))
    found = significance[significance["significant"].map(lambda v: v is True)]
    if len(found):
        print(f"\n  Significant differences (Holm-adjusted p < 0.05): {len(found)}")
        for r in found.itertuples():
            print(f"    {r.metric}: {r.approach_a} - {r.approach_b} = {r.difference:.3f} "
                  f"(p = {r.p_holm:.3f})")
    else:
        print("\n  No significant differences at Holm-adjusted p < 0.05.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", default="config/experiment_plan.yaml")
    parser.add_argument("--profile", help="Override the plan: YAML path, id@version, or id")
    parser.add_argument("--scenarios", type=int, nargs="+", help="Override the plan's scenarios")
    parser.add_argument("--approaches", nargs="+", help="Only these approaches (plan names)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the plan's seeds")
    parser.add_argument("--rows", type=int, help="Override the plan's rows per run")
    parser.add_argument("--out", default=str(DATA / "runs"), help="Where batches are written")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan) if Path(args.plan).is_absolute() else ROOT / args.plan
    plan = yaml.safe_load(plan_path.read_text())
    for needed in ("reference/reference.csv", "calibration/calibration_stats.yaml",
                   "evaluation/held_back_stats.yaml"):
        if not (DATA / needed).exists():
            print(f"Missing data/{needed}. Build the stores first with `make reference`.")
            return 1

    try:
        profile, profile_path = resolve_profile(args.profile or plan["profile"])
    except FileNotFoundError as exc:
        print(exc)
        return 1
    if not profile.is_approved:
        print(
            f"Profile {profile.profile_id} ({profile_path.relative_to(ROOT)}) is not approved, "
            "and only approved profiles drive generation. Approve it on the Review screen, "
            "or choose an approved one with --profile (for example "
            "--profile s01-civil-money-claims)."
        )
        return 1

    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    planned = plan_approaches(plan, models)
    if args.approaches:
        wanted = {PLAN_NAMES.get(n, n) for n in args.approaches}
        planned = [p for p in planned if p.approach.key in wanted]
    scenarios = args.scenarios or [s["id"] for s in plan["scenarios"]]
    seeds = args.seeds or list(plan["seeds"])
    n_rows = args.rows or int(plan["rows_per_run"])
    gateway = LLMGateway(models["ollama_url"])

    print(
        f"Experiment plan '{plan['name']}': profile {profile.profile_id}@{profile.version}, "
        f"scenarios {scenarios}, {len(planned)} approaches, seeds {seeds}, "
        f"{n_rows:,} rows per run"
    )
    folders = []
    for scenario in scenarios:
        print(f"\nScenario {scenario}")
        folders.append(
            run_scenario(scenario, planned, profile, n_rows, seeds, plan["name"], gateway,
                         Path(args.out))
        )
    in_app = Path(args.out).resolve() == (DATA / "runs").resolve()
    print("\nBatches" + (" (also listed on the Report screen):" if in_app else ":"))
    for f in folders:
        print(f"  {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
