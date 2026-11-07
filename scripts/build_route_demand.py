"""Clean the monthly passenger totals into one demand row per GTFS bus route.

Meaning of the source field (portal data dictionary): TRIPS = "Number of Passengers (rides)",
reported per YEAR, MONTH, ROUTE_NAME. This is ROUTE-LEVEL demand, not passengers per road segment.

Inputs : data/raw/monthly_totals/*.csv, data/interim/gtfs/feed/routes.txt,
         data/processed/route_summary.csv (optional, adds share_in_dubai)
Outputs: data/processed/route_demand.csv        one row per GTFS bus route
         data/processed/demand_unmatched.csv    monthly route names with no GTFS bus route
Nothing is dropped silently: every exclusion is counted and printed.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEAR = "2026"
MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]  # demand window (a setting, not a fixed rule)


def main(monthly_dir: Path, feed: Path, summary: Path, out_dir: Path, year: str, months: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(monthly_dir.glob("*.csv"))
    if len(files) != 1:
        raise SystemExit(f"Expected exactly one CSV in {monthly_dir}, found {len(files)}")
    m = pd.read_csv(files[0], dtype=str, encoding="utf-8-sig")
    print(f"monthly file: {files[0].name} | rows: {len(m):,}")

    w = m[(m["year"] == year) & (m["month"].isin(months))].copy()
    print(f"window {year} {months[0]}-{months[-1]}: {len(w):,} rows")
    n_null = int(w["route_name"].isna().sum())
    print(f"  rows with no route_name in window: {n_null}")
    w = w.dropna(subset=["route_name"])
    w["route_name"] = w["route_name"].str.strip()
    w["passengers"] = pd.to_numeric(w["trips"], errors="raise")

    key = ["year", "month", "route_name"]
    dup = w[w.duplicated(key, keep=False)]
    if len(dup):
        conflicts = int((dup.groupby(key)["passengers"].nunique() > 1).sum())
        print(f"  repeated keys in window: {dup.groupby(key).ngroups} (conflicting values: {conflicts})")
        if conflicts:
            raise SystemExit("Conflicting duplicate values in the window; decide a rule before continuing")
        w = w.drop_duplicates(key)
    else:
        print("  repeated keys in window: 0")

    # one row per route: wide table of months (a missing month is NOT assumed to be zero here)
    wide = w.pivot(index="route_name", columns="month", values="passengers").reindex(columns=months)
    agg = pd.DataFrame({
        "months_reported": wide.notna().sum(axis=1),
        "months_active": (wide.fillna(0) > 0).sum(axis=1),
        "window_total": wide.sum(axis=1),
        "mean_all_months": wide.fillna(0).sum(axis=1) / len(months),   # zeros/missing count as 0
        "mean_active_months": wide[wide > 0].mean(axis=1),            # only months with service
    })
    agg = agg.join(wide.add_prefix("m_"))

    # ---- crosswalk to GTFS bus routes (exact match on route_short_name) ----
    routes = pd.read_csv(feed / "routes.txt", dtype=str, encoding="utf-8-sig")
    bus = routes[routes["route_type"] == "3"][["route_id", "route_short_name"]]
    assert bus["route_short_name"].is_unique
    agg = agg.reset_index().rename(columns={"route_name": "route_short_name"})
    matched = bus.merge(agg, on="route_short_name", how="left")
    if summary.exists():
        rs = pd.read_csv(summary, dtype={"route_id": str})[["route_id", "share_in_dubai"]]
        matched = matched.merge(rs, on="route_id", how="left")
    matched["has_demand_row"] = matched["months_reported"].notna()

    unmatched = agg[~agg["route_short_name"].isin(bus["route_short_name"])]
    unmatched = unmatched[["route_short_name", "months_active", "window_total"]].sort_values("window_total", ascending=False)

    tot = agg["window_total"].sum()
    print(f"\nGTFS bus routes: {len(bus)} | with a demand row: {int(matched['has_demand_row'].sum())} "
          f"| without: {int((~matched['has_demand_row']).sum())}")
    print("GTFS bus routes without demand:", sorted(matched.loc[~matched["has_demand_row"], "route_short_name"]))
    print(f"monthly route names not in GTFS bus routes: {len(unmatched)} "
          f"| with passengers > 0: {int((unmatched['window_total'] > 0).sum())} "
          f"| share of window passengers: {unmatched['window_total'].sum() / tot:.2%}")
    print("largest unmatched:", unmatched.head(5).set_index("route_short_name")["window_total"].astype(int).to_dict())
    part = matched[matched["has_demand_row"] & (matched["months_active"] < len(months))]
    print(f"matched routes active in fewer than {len(months)} months: {len(part)}  "
          f"(mean_all_months vs mean_active_months differ for these)")
    print(f"matched routes with zero passengers across the window: {int((matched['window_total'] == 0).sum())}")
    print(f"window passengers matched: {matched['window_total'].sum():,.0f} of {tot:,.0f} "
          f"({matched['window_total'].sum() / tot:.2%})")

    matched.to_csv(out_dir / "route_demand.csv", index=False)
    unmatched.to_csv(out_dir / "demand_unmatched.csv", index=False)
    print("\nwrote route_demand.csv and demand_unmatched.csv in", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monthly-dir", type=Path, default=ROOT / "data/raw/monthly_totals")
    p.add_argument("--feed", type=Path, default=ROOT / "data/interim/gtfs/feed")
    p.add_argument("--summary", type=Path, default=ROOT / "data/processed/route_summary.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed")
    p.add_argument("--year", default=YEAR)
    p.add_argument("--months", nargs="+", default=MONTHS)
    a = p.parse_args()
    main(a.monthly_dir, a.feed, a.summary, a.out, a.year, a.months)
