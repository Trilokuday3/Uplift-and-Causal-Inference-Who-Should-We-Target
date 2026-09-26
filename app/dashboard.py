import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.data import ab_verdict, decile_frame, load_json, policy_at_budget, qini_frame  # noqa: E402

DOCS_DIR = os.environ.get("DOCS_DIR", str(ROOT / "docs"))

st.set_page_config(page_title="Uplift targeting", layout="wide")
st.title("Uplift & causal inference: who should we target?")
st.caption("Criteo Uplift v2.1. Streaming views are a simulated real-time replay, not live traffic.")

ab = load_json(DOCS_DIR, "ab_test_results.json")
uplift = load_json(DOCS_DIR, "uplift_results.json")
policy = load_json(DOCS_DIR, "policy_results.json")
streaming = load_json(DOCS_DIR, "streaming_results.json")

st.header("Did the campaign work?")
if ab:
    verdict = ab_verdict(ab)
    (st.success if verdict["worked"] else st.warning)(verdict["headline"])
    st.caption(f"{verdict['n_treated']:,} treated vs {verdict['n_control']:,} control users (randomized).")
else:
    st.warning("No A/B results found. Run `make ab-test`.")

st.header("Model ranking quality")
if uplift:
    st.subheader("Qini curves")
    st.line_chart(qini_frame(uplift))
    st.subheader(f"Observed uplift by predicted-uplift decile ({uplift['best_model']})")
    st.bar_chart(decile_frame(uplift).set_index("decile")["uplift"])
else:
    st.warning("No model results found. Run `make train`.")

st.header("Budget and business impact")
if policy:
    budget = st.slider("Share of users to target", 0.05, 1.0, 0.3, 0.05)
    assumptions = policy["assumptions"]
    rows = {}
    for strategy in policy["strategies"]:
        point = policy_at_budget(policy, budget, strategy)
        rows[strategy] = {"incremental outcomes": point["incremental_outcomes"], f"profit ({assumptions['currency']})": point["profit"]}
    st.table(pd.DataFrame(rows).T)
    st.caption(assumptions.get("note", ""))
else:
    st.warning("No policy results found. Run `make policy`.")

if streaming:
    st.header("Streaming replay: running ATE")
    st.line_chart(pd.DataFrame(streaming["running_ate"]).set_index("window_start")[["ate", "ci_low", "ci_high"]])
