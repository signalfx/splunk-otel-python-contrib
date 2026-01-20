"""Simulation runner with drift modes."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from validation import ValidationHarness
from main import run_scenario, save_artifacts


class DriftSimulator:
    """Simulates drift in tool responses and agent behavior."""

    def __init__(self, mode: str, intensity: float):
        """Initialize drift simulator.

        Args:
            mode: Drift mode ("tool_failure", "retriever_drift", "prompt_drift")
            intensity: Intensity of drift (0.0 to 1.0)
        """
        self.mode = mode
        self.intensity = intensity

    def apply_drift(self, config: Config):
        """Apply drift to configuration."""
        if self.mode == "tool_failure":
            # Simulate tool failures by adding error rate
            os.environ["TOOL_ERROR_RATE"] = str(self.intensity)
        elif self.mode == "retriever_drift":
            # Degrade retrieval by reducing top-k
            os.environ["RETRIEVER_TOP_K"] = str(int(3 * (1 - self.intensity)))
        elif self.mode == "prompt_drift":
            # Add noise to prompts (handled in agents)
            os.environ["PROMPT_NOISE_LEVEL"] = str(self.intensity)


def run_simulation(
    scenario_ids: List[str],
    iterations: int,
    drift_mode: Optional[str],
    drift_intensity: float,
    config: Config,
) -> Dict:
    """Run simulation over multiple iterations.

    Args:
        scenario_ids: List of scenario IDs to run
        iterations: Number of iterations per scenario
        drift_mode: Optional drift mode to apply
        drift_intensity: Intensity of drift
        config: Configuration

    Returns:
        Dict with simulation results
    """
    results = []
    validation_harness = ValidationHarness(config)

    drift_sim: Optional[DriftSimulator] = None
    if drift_mode:
        drift_sim = DriftSimulator(drift_mode, drift_intensity)

    for scenario_id in scenario_ids:
        for iteration in range(iterations):
            print(f"\n{'=' * 60}")
            print(f"Scenario: {scenario_id}, Iteration: {iteration + 1}/{iterations}")
            print(f"{'=' * 60}")

            # Apply drift if enabled
            if drift_mode:
                drift_sim.apply_drift(config)
                # Increase drift intensity over iterations (only if multiple iterations)
                if iterations > 1:
                    current_intensity = drift_intensity * (1 + iteration * 0.1)
                    drift_sim.intensity = min(current_intensity, 1.0)
                else:
                    drift_sim.intensity = drift_intensity

            try:
                # Run scenario
                config.scenario_id = scenario_id
                final_state = run_scenario(scenario_id, config)

                # Generate run ID
                run_id = f"{scenario_id}-iter{iteration + 1}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

                # Save artifacts
                save_artifacts(final_state, config, run_id)

                # Generate validation report (business logic only)
                # Evaluation metrics are handled automatically by opentelemetry-util-genai-evals
                validation_report = validation_harness.generate_validation_report(
                    final_state, run_id
                )

                # Save validation report
                artifacts_dir = Path(config.artifacts_dir) / run_id
                with open(artifacts_dir / "validation_report.json", "w") as f:
                    json.dump(validation_report, f, indent=2)

                # Collect results
                result = {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "iteration": iteration + 1,
                    "drift_mode": drift_mode,
                    "drift_intensity": drift_sim.intensity if drift_sim else 0.0,
                    "validation_passed": validation_report["validation_passed"],
                    "validations": validation_report["validations"],
                    "note": "Evaluation metrics (bias, toxicity, etc.) are emitted as OTEL metrics/logs automatically",
                }
                results.append(result)

                print(f"\n✅ Iteration {iteration + 1} completed")
                print(f"   Validation Passed: {validation_report['validation_passed']}")
                print(
                    f"   Hypothesis Match: {validation_report['validations']['hypothesis'].get('hypothesis_match', False)}"
                )
                print(
                    f"   Evidence Sufficient: {validation_report['validations']['evidence'].get('evidence_sufficient', False)}"
                )

            except Exception as e:
                print(f"\n❌ Error in iteration {iteration + 1}: {e}")
                results.append(
                    {
                        "scenario_id": scenario_id,
                        "iteration": iteration + 1,
                        "error": str(e),
                        "validation_passed": False,
                    }
                )

    # Generate summary
    summary = {
        "total_runs": len(results),
        "passed_runs": sum(1 for r in results if r.get("validation_passed", False)),
        "failed_runs": sum(1 for r in results if not r.get("validation_passed", True)),
        "drift_mode": drift_mode,
        "results": results,
        "note": "Evaluation metrics (bias, toxicity, relevance, etc.) are emitted automatically as OTEL metrics/logs during runs",
    }

    return summary


def main():
    """Main entry point for simulation runner."""
    parser = argparse.ArgumentParser(
        description="SRE Incident Copilot Simulation Runner"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["scenario-001"],
        help="Scenario IDs to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per scenario (default: 1, use 1 for single run with drift)",
    )
    parser.add_argument(
        "--drift-mode",
        type=str,
        choices=["tool_failure", "retriever_drift", "prompt_drift"],
        help="Drift mode to apply",
    )
    parser.add_argument(
        "--drift-intensity",
        type=float,
        default=0.1,
        help="Initial drift intensity (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for simulation results (JSON)",
    )
    args = parser.parse_args()

    config = Config.from_env()
    config.drift_enabled = args.drift_mode is not None
    config.drift_mode = args.drift_mode
    config.drift_intensity = args.drift_intensity

    print("🔄 SRE Incident Copilot Simulation Runner")
    print("=" * 60)
    print(f"Scenarios: {args.scenarios}")
    print(f"Iterations: {args.iterations}")
    if args.drift_mode:
        print(f"Drift Mode: {args.drift_mode}")
        print(f"Drift Intensity: {args.drift_intensity}")
    print()

    # Run simulation
    summary = run_simulation(
        args.scenarios,
        args.iterations,
        args.drift_mode,
        args.drift_intensity,
        config,
    )

    # Save summary
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📊 Summary saved to: {args.output}")
    else:
        summary_file = f"simulation_summary_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📊 Summary saved to: {summary_file}")

    # Print summary
    print("\n📈 Simulation Summary")
    print(f"   Total Runs: {summary['total_runs']}")
    print(f"   Passed: {summary['passed_runs']}")
    print(f"   Failed: {summary['failed_runs']}")
    print("\nNote: Evaluation metrics are emitted as OTEL metrics/logs automatically")


if __name__ == "__main__":
    main()
