"""Quick test script for Phase 2 modules."""
import sys
sys.path.insert(0, ".")

from src.data_generator.reference_data import (
    get_drug_reference_df, get_brand_generic_pairs,
    DRUG_DIAGNOSIS_MAP, CONDITION_DRUG_CLASSES,
)
from src.data_generator.entities import (
    generate_pharmacies, generate_prescribers, generate_members,
)
from src.data_generator.claims import generate_claims

# --- Reference Data ---
print("=" * 50)
print("REFERENCE DATA")
print("=" * 50)
df = get_drug_reference_df()
print(f"Total drugs: {len(df)}")
print(f"Therapeutic classes: {df.therapeutic_class.nunique()}")
print(f"DEA scheduled: {len(df[df.dea_schedule != 'NONE'])}")
print(f"Brand/Generic: {df.is_brand.sum()} / {(~df.is_brand).sum()}")

# --- Entities ---
print("\n" + "=" * 50)
print("ENTITIES")
print("=" * 50)
pharma = generate_pharmacies(500)
print(f"Pharmacies: {len(pharma)}")
print(f"  Suspicious: {pharma.is_suspicious.sum()} ({pharma.is_suspicious.mean()*100:.1f}%)")
print(f"  Types: {dict(pharma.pharmacy_type.value_counts())}")

prescribers = generate_prescribers(2000)
print(f"Prescribers: {len(prescribers)}")
print(f"  Suspicious: {prescribers.is_suspicious.sum()} ({prescribers.is_suspicious.mean()*100:.1f}%)")

members = generate_members(10000)
print(f"Members: {len(members)}")
print(f"  Doctor shoppers: {members.is_doctor_shopper.sum()} ({members.is_doctor_shopper.mean()*100:.1f}%)")
print(f"  Avg age: {members.age.mean():.1f}")
print(f"  Avg conditions: {members.num_conditions.mean():.1f}")
print(f"  Plans: {dict(members.plan_type.value_counts())}")

# --- Claims (small test: 5000) ---
print("\n" + "=" * 50)
print("CLAIMS (5,000 test batch)")
print("=" * 50)
claims = generate_claims(members, pharma, prescribers, num_claims=5000)
print(f"Claims generated: {len(claims)}")
print(f"  Paid: {(claims.claim_status == 'paid').sum()}")
print(f"  Rejected: {(claims.claim_status == 'rejected').sum()}")
print(f"  Reversed: {(claims.claim_status == 'reversed').sum()}")
print(f"  Avg total cost: ${claims.total_cost.mean():.2f}")
print(f"  Median total cost: ${claims.total_cost.median():.2f}")
print(f"  Unique members: {claims.member_id.nunique()}")
print(f"  Unique prescribers: {claims.prescriber_id.nunique()}")
print(f"  Unique pharmacies: {claims.pharmacy_id.nunique()}")
print(f"  Drug classes: {claims.therapeutic_class.nunique()}")
print(f"  DEA scheduled claims: {(claims.dea_schedule != 'NONE').sum()}")
print(f"  Prior auth: {claims.prior_auth_flag.sum()}")
print(f"  Weekend claims: {(claims.service_date.dt.weekday >= 5).sum()}")
print(f"  Business hours (8-18): {((claims.submit_hour >= 8) & (claims.submit_hour <= 18)).sum()}")

print("\nTop 5 drug classes:")
for tc, cnt in claims.therapeutic_class.value_counts().head(5).items():
    print(f"  {tc}: {cnt}")

print("\n" + "=" * 50)
print("ALL PHASE 2 TESTS PASSED!")
print("=" * 50)
