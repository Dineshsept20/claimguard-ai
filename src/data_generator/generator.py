"""Master data generator orchestrator.

Produces the complete synthetic pharmacy claims dataset by:
1. Generating entities (pharmacies, prescribers, members)
2. Generating baseline normal claims
3. Injecting anomalies (7 fraud patterns)
4. Validating data integrity
5. Saving to data/raw/ as CSV files

CLI usage:
    python -m src.data_generator.generator --num-claims 500000
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from .anomalies import inject_anomalies
from .claims import generate_claims
from .entities import generate_members, generate_pharmacies, generate_prescribers


def generate_dataset(
    num_claims: int = 500_000,
    num_pharmacies: int = 500,
    num_prescribers: int = 2000,
    num_members: int = 10_000,
    anomaly_rate: float = 0.04,
    output_dir: str = "data/raw",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate the complete ClaimGuard AI dataset.

    Args:
        num_claims: Total number of claims to generate.
        num_pharmacies: Number of pharmacies.
        num_prescribers: Number of prescribers.
        num_members: Number of members/patients.
        anomaly_rate: Fraction of claims to inject as anomalies (0.03–0.05).
        output_dir: Directory to save CSV files.
        seed: Master random seed.

    Returns:
        Dictionary with keys 'claims', 'pharmacies', 'prescribers', 'members'.
    """
    overall_start = time.time()
    print("=" * 60)
    print("ClaimGuard AI — Synthetic Data Generator")
    print("=" * 60)
    print(f"  Target claims:  {num_claims:,}")
    print(f"  Pharmacies:     {num_pharmacies:,}")
    print(f"  Prescribers:    {num_prescribers:,}")
    print(f"  Members:        {num_members:,}")
    print(f"  Anomaly rate:   {anomaly_rate*100:.1f}%")
    print(f"  Seed:           {seed}")
    print()

    # ---- Step 1: Generate Entities ----
    print("[1/5] Generating entities...")
    t0 = time.time()
    pharmacies = generate_pharmacies(num_pharmacies, seed=seed)
    prescribers = generate_prescribers(num_prescribers, seed=seed + 1)
    members = generate_members(num_members, seed=seed + 2)
    print(f"  Entities generated in {time.time()-t0:.1f}s")
    print(f"    Pharmacies:  {len(pharmacies):,} ({pharmacies.is_suspicious.sum()} suspicious)")
    print(f"    Prescribers: {len(prescribers):,} ({prescribers.is_suspicious.sum()} suspicious)")
    print(f"    Members:     {len(members):,} ({members.is_doctor_shopper.sum()} doctor shoppers)")
    print()

    # ---- Step 2: Generate Normal Claims ----
    print("[2/5] Generating normal claims...")
    t0 = time.time()
    claims = generate_claims(
        members, pharmacies, prescribers,
        num_claims=num_claims,
        seed=seed + 3,
    )
    print(f"  Claims generated in {time.time()-t0:.1f}s")
    print()

    # ---- Step 3: Inject Anomalies ----
    print("[3/5] Injecting anomalies...")
    t0 = time.time()
    claims = inject_anomalies(
        claims, members, pharmacies, prescribers,
        anomaly_rate=anomaly_rate,
        seed=seed + 4,
    )
    print(f"  Anomalies injected in {time.time()-t0:.1f}s")
    print()

    # ---- Step 4: Validate Data ----
    print("[4/5] Validating data integrity...")
    _validate_dataset(claims, pharmacies, prescribers, members)
    print()

    # ---- Step 5: Save to CSV ----
    print("[5/5] Saving to CSV...")
    os.makedirs(output_dir, exist_ok=True)

    claims_path = os.path.join(output_dir, "claims.csv")
    pharmacies_path = os.path.join(output_dir, "pharmacies.csv")
    prescribers_path = os.path.join(output_dir, "prescribers.csv")
    members_path = os.path.join(output_dir, "members.csv")

    claims.to_csv(claims_path, index=False)
    pharmacies.to_csv(pharmacies_path, index=False)
    prescribers.to_csv(prescribers_path, index=False)
    members.to_csv(members_path, index=False)

    # File sizes
    for path, name in [
        (claims_path, "claims.csv"),
        (pharmacies_path, "pharmacies.csv"),
        (prescribers_path, "prescribers.csv"),
        (members_path, "members.csv"),
    ]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB")

    elapsed = time.time() - overall_start
    print()
    print("=" * 60)
    print(f"Dataset generation complete in {elapsed:.1f}s")
    print("=" * 60)

    return {
        "claims": claims,
        "pharmacies": pharmacies,
        "prescribers": prescribers,
        "members": members,
    }


def _validate_dataset(
    claims: pd.DataFrame,
    pharmacies: pd.DataFrame,
    prescribers: pd.DataFrame,
    members: pd.DataFrame,
) -> None:
    """Validate the generated dataset for integrity and realism."""
    errors = []
    warnings = []

    # 1. Foreign key checks
    invalid_pharmacies = ~claims["pharmacy_id"].isin(pharmacies["pharmacy_id"])
    if invalid_pharmacies.any():
        errors.append(f"  ❌ {invalid_pharmacies.sum()} claims reference invalid pharmacy_id")

    invalid_prescribers = ~claims["prescriber_id"].isin(prescribers["prescriber_id"])
    if invalid_prescribers.any():
        errors.append(f"  ❌ {invalid_prescribers.sum()} claims reference invalid prescriber_id")

    invalid_members = ~claims["member_id"].isin(members["member_id"])
    if invalid_members.any():
        errors.append(f"  ❌ {invalid_members.sum()} claims reference invalid member_id")

    # 2. Anomaly rate check
    anomaly_rate = claims["is_anomaly"].mean()
    if anomaly_rate < 0.01 or anomaly_rate > 0.10:
        warnings.append(f"  ⚠️  Anomaly rate {anomaly_rate*100:.2f}% outside expected range (1-10%)")

    # 3. Cost sanity checks
    negative_costs = (claims["total_cost"] < 0).sum()
    if negative_costs > 0:
        errors.append(f"  ❌ {negative_costs} claims with negative total_cost")

    zero_costs = (claims["total_cost"] == 0).sum()
    if zero_costs > len(claims) * 0.01:
        warnings.append(f"  ⚠️  {zero_costs} claims with $0 total cost")

    # 4. Date range check
    min_date = claims["service_date"].min()
    max_date = claims["service_date"].max()
    if pd.Timestamp("2025-01-01") > min_date or pd.Timestamp("2026-01-01") < max_date:
        warnings.append(f"  ⚠️  Date range {min_date} to {max_date} outside expected 2025")

    # 5. Claim status distribution
    paid_pct = (claims["claim_status"] == "paid").mean()
    if paid_pct < 0.85 or paid_pct > 0.98:
        warnings.append(f"  ⚠️  Paid claims {paid_pct*100:.1f}% outside expected range (85-98%)")

    # 6. Weekend distribution (should be reduced vs 28%)
    weekend_pct = (claims["service_date"].dt.weekday >= 5).mean()
    if weekend_pct > 0.25:
        warnings.append(f"  ⚠️  Weekend claims {weekend_pct*100:.1f}% higher than expected (<25%)")

    # 7. Unique claim IDs
    if claims["claim_id"].nunique() != len(claims):
        errors.append(f"  ❌ Duplicate claim_id values found")

    # Report
    if errors:
        print("  VALIDATION ERRORS:")
        for e in errors:
            print(e)
    if warnings:
        print("  VALIDATION WARNINGS:")
        for w in warnings:
            print(w)
    if not errors and not warnings:
        print("  ✅ All validation checks passed")
    elif not errors:
        print("  ✅ No critical errors (warnings only)")


def main():
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="ClaimGuard AI — Synthetic Pharmacy Claims Data Generator"
    )
    parser.add_argument(
        "--num-claims", type=int, default=500_000,
        help="Number of claims to generate (default: 500000)"
    )
    parser.add_argument(
        "--num-pharmacies", type=int, default=500,
        help="Number of pharmacies (default: 500)"
    )
    parser.add_argument(
        "--num-prescribers", type=int, default=2000,
        help="Number of prescribers (default: 2000)"
    )
    parser.add_argument(
        "--num-members", type=int, default=10_000,
        help="Number of members/patients (default: 10000)"
    )
    parser.add_argument(
        "--anomaly-rate", type=float, default=0.04,
        help="Fraction of claims that are anomalous (default: 0.04)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Output directory for CSV files (default: data/raw)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    generate_dataset(
        num_claims=args.num_claims,
        num_pharmacies=args.num_pharmacies,
        num_prescribers=args.num_prescribers,
        num_members=args.num_members,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
