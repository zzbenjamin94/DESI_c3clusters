"""Validate the real DR2 BGS data and random catalogs on NERSC.

The default mode uses memory-mapped FITS access and samples rows spread across
each file. Use ``--full-data-scan`` to scan every BGS row after the sampled
checks pass.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.units import UnitsWarning


DEFAULT_BGS_DIR = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1"
)
DEFAULT_BGS_FILE = DEFAULT_BGS_DIR / "BGS_BRIGHT_full_noveto.dat.fits"
DEFAULT_RANDOM_PATTERN = "BGS_BRIGHT_*_full.ran.fits"

REQUIRED_DATA_COLUMNS = {
    "TARGETID",
    "RA",
    "DEC",
    "Z",
    "PROB_OBS",
    "FLUX_R",
}
REQUIRED_RANDOM_COLUMNS = {"RA", "DEC"}


class ValidationReport:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        label = "PASS" if condition else "FAIL"
        print(f"[{label}] {message}")
        if not condition:
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if condition:
            print(f"[PASS] {message}")
        else:
            print(f"[WARN] {message}")
            self.warnings.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgs-file", type=Path, default=DEFAULT_BGS_FILE)
    parser.add_argument("--random-dir", type=Path, default=DEFAULT_BGS_DIR)
    parser.add_argument("--random-pattern", default=DEFAULT_RANDOM_PATTERN)
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--max-random-files", type=int, default=3)
    parser.add_argument(
        "--full-data-scan",
        action="store_true",
        help="Inspect all BGS rows after the sampled checks.",
    )
    return parser.parse_args()


def table_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return hdu
    raise ValueError("FITS file contains no table HDU")


def spread_indices(n_rows: int, sample_rows: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=int)
    n_sample = min(n_rows, max(1, sample_rows))
    return np.linspace(0, n_rows - 1, n_sample, dtype=int)


def finite_percentiles(values, percentiles=(0, 1, 16, 50, 84, 99, 100)):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.full(len(percentiles), np.nan)
    return np.nanpercentile(finite, percentiles)


def summarize_data_rows(rows, report: ValidationReport, label: str) -> None:
    n = len(rows)
    ra = np.asarray(rows["RA"], dtype=float)
    dec = np.asarray(rows["DEC"], dtype=float)
    redshift = np.asarray(rows["Z"], dtype=float)
    prob_obs = np.asarray(rows["PROB_OBS"], dtype=float)
    flux_r = np.asarray(rows["FLUX_R"], dtype=float)
    targetid = np.asarray(rows["TARGETID"])

    valid_coord = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & (ra >= 0.0)
        & (ra < 360.0)
        & (dec >= -90.0)
        & (dec <= 90.0)
    )
    valid_prob = np.isfinite(prob_obs) & (prob_obs > 0.0) & (prob_obs <= 1.0)
    valid_flux = np.isfinite(flux_r) & (flux_r > 0.0)
    plausible_z = np.isfinite(redshift) & (redshift > 0.0) & (redshift < 1.0)
    sentinel_z = (~np.isfinite(redshift)) | (redshift >= 1.0)

    print(f"\n{label}")
    print(f"Rows inspected: {n:,}")
    print(f"Valid coordinates: {np.count_nonzero(valid_coord):,} ({np.mean(valid_coord):.4%})")
    print(f"Plausible 0<Z<1: {np.count_nonzero(plausible_z):,} ({np.mean(plausible_z):.4%})")
    print(f"Sentinel/non-analysis Z: {np.count_nonzero(sentinel_z):,} ({np.mean(sentinel_z):.4%})")
    print(f"Valid 0<PROB_OBS<=1: {np.count_nonzero(valid_prob):,} ({np.mean(valid_prob):.4%})")
    print(f"Positive finite FLUX_R: {np.count_nonzero(valid_flux):,} ({np.mean(valid_flux):.4%})")
    print("Z percentiles:", finite_percentiles(redshift))
    print("PROB_OBS percentiles:", finite_percentiles(prob_obs))
    print("FLUX_R percentiles:", finite_percentiles(flux_r))

    analysis_rows = valid_coord & plausible_z & valid_prob & valid_flux
    report.check(np.all(valid_coord), f"{label}: every inspected row has valid RA/DEC")
    report.check(np.count_nonzero(analysis_rows) > 0, f"{label}: at least one row passes basic analysis cuts")
    report.check(
        np.all((prob_obs[valid_prob] > 0.0) & (prob_obs[valid_prob] <= 1.0)),
        f"{label}: valid PROB_OBS values lie in (0, 1]",
    )
    report.warn(
        len(np.unique(targetid)) == len(targetid),
        f"{label}: inspected TARGETID values are unique",
    )


def validate_bgs_file(path: Path, sample_rows: int, full_scan: bool, report: ValidationReport) -> None:
    report.check(path.exists(), f"BGS data file exists: {path}")
    if not path.exists():
        return

    with fits.open(path, memmap=True) as hdul:
        hdu = table_hdu(hdul)
        names = set(hdu.columns.names)
        missing = sorted(REQUIRED_DATA_COLUMNS - names)
        report.check(not missing, f"BGS data contains required columns; missing={missing}")
        if missing:
            return

        n_rows = len(hdu.data)
        report.check(n_rows > 0, f"BGS data table is nonempty ({n_rows:,} rows)")
        indices = spread_indices(n_rows, sample_rows)
        summarize_data_rows(hdu.data[indices], report, "Spread sample")

        if full_scan:
            summarize_data_rows(hdu.data, report, "Full BGS scan")


def validate_random_files(
    random_dir: Path,
    pattern: str,
    max_files: int,
    sample_rows: int,
    report: ValidationReport,
) -> None:
    paths = sorted(random_dir.glob(pattern))
    report.check(len(paths) > 0, f"Discovered random catalogs matching {random_dir / pattern}")
    if not paths:
        return

    selected = paths if max_files <= 0 else paths[:max_files]
    print(f"Random catalogs discovered: {len(paths):,}; inspecting: {len(selected):,}")
    row_counts = []

    for path in selected:
        with fits.open(path, memmap=True) as hdul:
            hdu = table_hdu(hdul)
            names = set(hdu.columns.names)
            missing = sorted(REQUIRED_RANDOM_COLUMNS - names)
            report.check(not missing, f"{path.name}: required random columns; missing={missing}")
            if missing:
                continue

            n_rows = len(hdu.data)
            row_counts.append(n_rows)
            indices = spread_indices(n_rows, min(sample_rows, 20_000))
            rows = hdu.data[indices]
            ra = np.asarray(rows["RA"], dtype=float)
            dec = np.asarray(rows["DEC"], dtype=float)
            valid = (
                np.isfinite(ra)
                & np.isfinite(dec)
                & (ra >= 0.0)
                & (ra < 360.0)
                & (dec >= -90.0)
                & (dec <= 90.0)
            )
            report.check(np.all(valid), f"{path.name}: sampled random coordinates are valid")
            print(f"  {path.name}: {n_rows:,} rows")

    if len(row_counts) > 1:
        relative_spread = (max(row_counts) - min(row_counts)) / np.mean(row_counts)
        report.warn(
            relative_spread < 0.05,
            f"Inspected random-file row counts agree within 5% (spread={relative_spread:.3%})",
        )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UnitsWarning)
    report = ValidationReport()

    print("DR2 BGS catalog validation")
    print("BGS data:", args.bgs_file)
    print("Random directory:", args.random_dir)
    print("Random pattern:", args.random_pattern)

    validate_bgs_file(args.bgs_file, args.sample_rows, args.full_data_scan, report)
    validate_random_files(
        args.random_dir,
        args.random_pattern,
        args.max_random_files,
        args.sample_rows,
        report,
    )

    print("\nValidation summary")
    print(f"Failures: {len(report.failures)}")
    print(f"Warnings: {len(report.warnings)}")
    for message in report.failures:
        print(f"  FAIL: {message}")
    for message in report.warnings:
        print(f"  WARN: {message}")

    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
