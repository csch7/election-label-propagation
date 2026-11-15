"""County-level election labels (Democratic vs Republican winner)."""
import pandas as pd

from ..config import load_config
from ..graphs.graph_io import RAW_DIR


def build_labels(year=None, path=RAW_DIR / "Voting Data.csv"):
    """Return a Series indexed by 5-digit GEOID (year defaults to `target_year`
    in the config): 1 if the Democratic
    candidate won the county's presidential vote, 0 if the Republican did.

    Counties report votes split by mode (early, absentee, ...). Where a
    TOTAL row exists it is used, otherwise all modes are summed.
    """
    year = load_config()["target_year"] if year is None else year
    v = pd.read_csv(path, dtype={"GEOID": str})
    v = v[(v.year == year) & v.party.isin(["DEMOCRAT", "REPUBLICAN"])].copy()
    v["GEOID"] = v.GEOID.str.zfill(5)
    has_total = v.groupby("GEOID")["mode"].transform(lambda m: (m == "TOTAL").any())
    v = v[~has_total | (v["mode"] == "TOTAL")]
    votes = v.pivot_table(index="GEOID", columns="party", values="candidatevotes", aggfunc="sum").dropna()
    return (votes["DEMOCRAT"] > votes["REPUBLICAN"]).astype(int).rename("label")
