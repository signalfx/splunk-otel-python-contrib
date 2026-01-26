"""Simulation runner with drift modes.

Usage:
    python simulation_runner.py --scenarios scenario-001 scenario-002 --iterations 5

    # With zero-code instrumentation
    opentelemetry-instrument python simulation_runner.py --scenarios scenario-001 --iterations 3
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from main import run_scenario, save_artifacts, _setup_instrumentation
from validation import ValidationHarness


class DriftSimulator:
    """Simulates drift in tool responses and agent behavior."""

    def __init__(self, mode: str, intensity: float):
        self.mode = mode
        self.intensity = intensity

    def apply_drift(self, config: Config):
        """Apply drift to configuration."""
        if self.mode == "tool_failure":
            os.environ["TOOL_ERROR_RATE"] = str(self.intensity)
        elif self.mode == "retriever_drift":
            os.environ["RETRIEVER_TOP_K"] = str(int(3 * (1 - self.intensity)))
        elif self.mode == "prompt_drift":
            os.environ["PROMPT_NOISE_LEVEL"] = str(self.intensity)


def run_simulation(
    scenario_ids: List[str],
    iterations: int,
    drift_mode: Optional[str],
    drift_intensity: float,
    config: Config,
) -> Dict:
    """Run simulation over multiple iterations."""
    results = []
    validation_harness = ValidationHarness(config)

    drift_sim: Optional[DriftSimulator] = None
    if drift_mode:
        drift_sim = DriftSimulator(drift_mode, drift_intensity)

    for scenario_id in scenario_ids:
        for iteration in range(iterations):
            if drift_mode and drift_sim:
                drift_sim.apply_drift(config)
                if iterations > 1:
                    current_intensity = drift_intensity * (1 + iteration * 0.1)
                    drift_sim.intensity = min(current_intensity, 1.0)
                else:
                    drift_sim.intensity = drift_intensity

            try:
                config.scenario_id = scenario_id
                final_state = run_scenario(scenario_id, config)

                run_id = f"{scenario_id}-iter{iteration + 1}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                save_artifacts(final_state, config, run_id)

                validation_report = validation_harness.generate_validation_report(
                    final_state, run_id
                )

                artifacts_dir = Path(config.artifacts_dir) / run_id
                with open(artifacts_dir / "validation_report.json", "w") as f:
                    json.dump(validation_report, f, indent=2)

                result = {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "iteration": iteration + 1,
                    "drift_mode": drift_mode,
                    "drift_intensity": drift_sim.intensity if drift_sim else 0.0,
                    "validation_passed": validation_report["validation_passed"],
                    "validations": validation_report["validations"],
                }
                results.append(result)

                status = "✓" if validation_report['validation_passed'] else "✗"
                print(f"  {status} {scenario_id} iter {iteration + 1}", flush=True)

            except Exception as e:
                print(f"  ✗ {scenario_id} iter {iteration + 1}: {e}", flush=True)
                results.append({
                    "scenario_id": scenario_id,
                    "iteration": iteration + 1,
                    "error": str(e),
                    "validation_passed": False,
                })

    return {
        "total_runs": len(results),
        "passed_runs": sum(1 for r in results if r.get("validation_passed", False)),
        "failed_runs": sum(1 for r in results if not r.get("validation_passed", True)),
        "drift_mode": drift_mode,
        "results": results,
    }


def main():
    """Main entry point for simulation runner."""
    parser = argparse.ArgumentParser(description="SRE Incident Copilot Simulation Runner")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["scenario-001"], help="Scenario IDs to run")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations per scenario")
    parser.add_argument("--drift-mode", type=str, choices=["tool_failure", "retriever_drift", "prompt_drift"])
    parser.add_argument("--drift-intensity", type=float, default=0.1, help="Drift intensity (0.0 to 1.0)")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    args = parser.parse_args()

    # Setup instrumentation
    _setup_instrumentation()

    config = Config.from_env()
    config.drift_enabled = args.drift_mode is not None
    config.drift_mode = args.drift_mode
    config.drift_intensity = args.drift_intensity

    drift_info = f" | drift: {args.drift_mode}" if args.drift_mode else ""
    print(f"🔄 Simulation | {len(args.scenarios)} scenarios × {args.iterations} iterations{drift_info}", flush=True)

    summary = run_simulation(
        args.scenarios,
        args.iterations,
        args.drift_mode,
        args.drift_intensity,
        config,
    )

    # Save summary
    output_file = args.output or f"simulation_summary_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 {summary['passed_runs']}/{summary['total_runs']} passed | {output_file}", flush=True)


if __name__ == "__main__":
    main()
