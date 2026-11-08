"""Build the road-bus coupling tables from the matched bus lines.

Reads  data/processed/bus_route_segments.csv, data/processed/bus_match_quality.csv
       data/processed/route_demand.csv (only to report how much demand sits on high-coverage routes)
Writes data/processed/route_segments.csv   (route, segment_id, travel_dirs, n_lines, origin)
       data/processed/route_coverage.csv   (one row per route, with high_coverage flag)
"""
from pathlib import Path

import pandas as pd

from meteoric.route_coupling import HIGH_COVERAGE, route_coverage, route_segment_table

SEQ = Path("data/processed/bus_route_segments.csv")
QUAL = Path("data/processed/bus_match_quality.csv")
DEMAND = Path("data/processed/route_demand.csv")
OUT_SEG = Path("data/processed/route_segments.csv")
OUT_COV = Path("data/processed/route_coverage.csv")


def main() -> None:
    seq = pd.read_csv(SEQ, dtype={"route": str})
    qual = pd.read_csv(QUAL, dtype={"route": str})

    table = route_segment_table(seq)
    cov = route_coverage(qual, seq)
    print(f"Routes: {len(cov):,}   route-segment links: {len(table):,}   distinct segments used: {table['segment_id'].nunique():,}")
    print(f"Directions used: {table['travel_dirs'].value_counts().to_dict()}")
    print(f"Links that came from gap-filling only: {(table['origin'] == 'filled').sum():,}")

    print("\nCoverage (share of a route's path matched onto roads):")
    print(cov["coverage"].describe(percentiles=[0.05, 0.25, 0.5, 0.75]).round(3).to_string())
    print(f"\nHigh coverage (>= {HIGH_COVERAGE:.0%}): {int(cov['high_coverage'].sum()):,} of {len(cov):,} routes")
    print(f"Routes under 80% coverage: {int((cov['coverage'] < 0.8).sum()):,}")
    print("\nLowest-coverage routes:")
    print(cov.sort_values("coverage").head(10)[["route", "n_lines", "coverage", "unbridged_gaps", "n_segments"]]
          .round(3).to_string(index=False))

    if DEMAND.exists():
        dem = pd.read_csv(DEMAND, dtype=str)
        rcol = next((c for c in ("route_short_name", "route") if c in dem.columns), None)
        if rcol and "mean_all_months" in dem.columns:
            dem["demand"] = pd.to_numeric(dem["mean_all_months"], errors="coerce")
            m = cov.merge(dem[[rcol, "demand"]].rename(columns={rcol: "route"}), on="route", how="left")
            total = m["demand"].sum()
            print(f"\nDemand on high-coverage routes: {m.loc[m['high_coverage'], 'demand'].sum() / total:.1%} of total")
            print(f"Routes with no demand figure: {int(m['demand'].isna().sum())}")
        else:
            print(f"\n(Demand columns not recognised in {DEMAND}: {list(dem.columns)})")

    table.to_csv(OUT_SEG, index=False)
    cov.to_csv(OUT_COV, index=False)
    print(f"\nSaved {OUT_SEG} and {OUT_COV}")


if __name__ == "__main__":
    main()
