# Adapted from code by Maud Tissot (Spinup-NEMO)
# Original source: https://github.com/maudtst/Spinup-NEMO
# Licensed under the MIT License
#
# Modifications in this version by ICCS, 2025
import argparse
import os
from pathlib import Path

from nemo_spinup_forecast.dimensionality_reduction import (
    dimensionality_reduction_techniques,
)
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.pipeline import run_pipeline
from nemo_spinup_forecast.utils import (
    create_run_dir,
    get_dr_technique,
    get_forecast_technique,
    load_ocean_terms,
)


def main(argv=None) -> int:
    """Entry point for the emulator CLI."""
    parser = argparse.ArgumentParser(description="Emulator")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to simulation data to forecast from",
    )
    parser.add_argument(
        "--ye",
        type=bool,
        help="Transform monthly simulation to yearly simulation",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start of the training (0 to keep spin up / t to cut the spin up)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End of the training (end-start = train len)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of steps to emulate (years to forecast)",
    )
    parser.add_argument(
        "--comp",
        type=str,
        default="None",
        help="Explained variance ratio for the PCA (int, float, or 'None')",
    )
    parser.add_argument(
        "--ocean-terms",
        type=str,
        default=None,
        help="Path to ocean_terms.yaml (overrides the packaged default)",
    )
    parser.add_argument(
        "--techniques-config",
        type=str,
        default=None,
        help="Path to techniques_config.yaml (overrides package default)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to write forecast results to",
    )

    args = parser.parse_args(argv)

    # Validate that the data path exists
    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        parser.error(
            f"Data path does not exist: {args.data_path}\n"
            "  Please check the path and try again.\n"
        )
    if not data_path.is_dir():
        parser.error(
            f"Data path is not a directory: {args.data_path}\n"
            "  Please provide a path to the directory containing the simulation files."
        )

    # Load config file of techniques
    if args.techniques_config:
        techniques_config_path = Path(args.techniques_config).expanduser().resolve()
    else:
        techniques_config_path = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "configs/techniques_config.yaml"
        )

    dr_technique = get_dr_technique(
        techniques_config_path, dimensionality_reduction_techniques
    )
    forecast_technique = get_forecast_technique(
        techniques_config_path, forecast_techniques
    )

    # Create a per-run directory to store results
    run_dir = create_run_dir(args.output_path)
    out_dir = run_dir / "forecast"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert comp to int or float if possible
    if args.comp is not None:
        if args.comp.isdigit():
            args.comp = int(args.comp)
        elif args.comp.replace(".", "", 1).isdigit():
            args.comp = float(args.comp)
        elif args.comp == "None":
            args.comp = None

    ocean_terms_path: Path | None = (
        Path(args.ocean_terms).expanduser().resolve() if args.ocean_terms else None
    )
    specs = load_ocean_terms(ocean_terms_path)

    run_pipeline(
        specs,
        data_path=args.data_path,
        out_dir=out_dir,
        start=args.start,
        end=args.end,
        steps=args.steps,
        ye=args.ye,
        comp=args.comp,
        dr_technique=dr_technique,
        forecast_technique=forecast_technique,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
