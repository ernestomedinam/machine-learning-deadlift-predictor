"""Command-line inference app for the trained deadlift regression model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deadlift_model.joblib"


def load_artifact(model_path: Path) -> dict[str, Any]:
	"""Load the pipeline artifact produced by src/train.py."""
	if not model_path.exists():
		raise FileNotFoundError(
			f"Model artifact not found: {model_path}\n"
			"Train a model first with: python src/train.py"
		)

	artifact = joblib.load(model_path)
	if "pipeline" not in artifact or "feature_columns" not in artifact:
		raise ValueError(f"Unexpected model artifact format: {model_path}")
	return artifact


def predict_deadlift(model_path: Path, lifter_features: dict[str, Any]) -> float:
	"""Predict deadlift in kilograms from one lifter's original-scale features."""
	artifact = load_artifact(model_path)
	expected_features = artifact["feature_columns"]
	missing_features = [name for name in expected_features if name not in lifter_features]
	extra_features = [name for name in lifter_features if name not in expected_features]
	if missing_features:
		raise ValueError(f"Missing required input features: {missing_features}")
	if extra_features:
		raise ValueError(f"Unexpected input features: {extra_features}")

	input_frame = pd.DataFrame(
		[{name: lifter_features[name] for name in expected_features}],
		columns=expected_features,
	)
	prediction = artifact["pipeline"].predict(input_frame)[0]
	return float(prediction)


def _prompt_choice(prompt: str, allowed_values: list[str]) -> str:
	options = ", ".join(allowed_values)
	while True:
		value = input(f"{prompt} ({options}): ").strip()
		if value in allowed_values:
			return value
		print(f"Choose one of: {options}")


def _prompt_float(prompt: str, minimum: float = 0.0) -> float:
	while True:
		raw_value = input(f"{prompt}: ").strip()
		try:
			value = float(raw_value)
		except ValueError:
			print("Please enter a number.")
			continue
		if value < minimum:
			print(f"Please enter a value greater than or equal to {minimum}.")
			continue
		return value


def collect_lifter_features() -> dict[str, Any]:
	"""Prompt for all seven features used by the model."""
	print("Enter the lifter's data. Bodyweight and lift values are in kg.")
	return {
		"Sex": _prompt_choice("Sex", ["M", "F", "Mx"]),
		"Equipment": _prompt_choice(
			"Equipment", ["Raw", "Single-ply", "Multi-ply", "Straps", "Wraps", "Unlimited"]
		),
		"Age": _prompt_float("Age in years"),
		"BodyweightKg": _prompt_float("Bodyweight in kg"),
		"Event": _prompt_choice("Event", ["SBD", "BD", "D", "SD"]),
		"max_squat_kg": _prompt_float("Maximum squat in kg"),
		"max_bench_kg": _prompt_float("Maximum bench in kg"),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--model",
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help=f"Trained pipeline artifact (default: {DEFAULT_MODEL_PATH})",
	)
	parser.add_argument("--sex", choices=["M", "F", "Mx"])
	parser.add_argument(
		"--equipment",
		choices=["Raw", "Single-ply", "Multi-ply", "Straps", "Wraps", "Unlimited"],
	)
	parser.add_argument("--age", type=float)
	parser.add_argument("--bodyweight-kg", type=float)
	parser.add_argument("--event", choices=["SBD", "BD", "D", "SD"])
	parser.add_argument("--max-squat-kg", type=float)
	parser.add_argument("--max-bench-kg", type=float)
	args = parser.parse_args()

	cli_values = {
		"Sex": args.sex,
		"Equipment": args.equipment,
		"Age": args.age,
		"BodyweightKg": args.bodyweight_kg,
		"Event": args.event,
		"max_squat_kg": args.max_squat_kg,
		"max_bench_kg": args.max_bench_kg,
	}
	if all(value is None for value in cli_values.values()):
		lifter_features = collect_lifter_features()
	elif any(value is None for value in cli_values.values()):
		parser.error("Provide all seven feature flags, or provide none to use interactive prompts.")
	else:
		lifter_features = cli_values
		numeric_fields = ["Age", "BodyweightKg", "max_squat_kg", "max_bench_kg"]
		if any(lifter_features[field] < 0 for field in numeric_fields):
			parser.error("Age, bodyweight, squat, and bench values must be non-negative.")

	try:
		prediction = predict_deadlift(args.model, lifter_features)
	except (FileNotFoundError, ValueError) as error:
		parser.error(str(error))

	print(f"Predicted maximum deadlift: {prediction:.1f} kg")


if __name__ == "__main__":
	main()
