"""AI Strategic Advisory – Claude generates recommendations from Python-calculated evidence."""
import json

import pandas as pd
import streamlit as st

from src import ai_advisor as ai
from src import data_access as da
from src import database as db
from src import insights
from src import ui
from src.config import ANTHROPIC_MODEL

ui.header("AI Strategic Advisory", "Claude reads the evidence pack calculated by Python and proposes management actions. "
          "It does not calculate financials itself.")
if not da.data_status()["Loaded"].any():
    ui.require(["income"])

st.markdown(f"{ui.badge('FACT')} calculated from data · {ui.badge('FORECAST')} statistical model · "
            f"{ui.badge('SCENARIO')} assumption-based · {ui.badge('OPINION')} AI judgement")

with st.sidebar:
    st.markdown("**Claude API**")
    key_in = st.text_input("API key (optional if set in .env)", type="password", key="ai_key",
                           help="Used for this session only; not stored.")
    model = st.text_input("Model", ANTHROPIC_MODEL, key="ai_model")
api_key = ai.get_key(key_in)

c1, c2 = st.columns([2, 1])
areas = c1.multiselect("Recommendation areas", ai.AREAS, default=ai.AREAS, key="ai_areas")
include_fc = c2.toggle("Include forecasts in evidence (slower)", value=True, key="ai_fc")
focus = st.text_area("Additional management focus (optional)", key="ai_focus",
                     placeholder="e.g. Board wants to protect EBITDA margin above 18% while defending market share")

if st.button("Build evidence pack", key="ai_build") or "ai_ctx" not in st.session_state:
    with st.spinner("Calculating evidence pack…"):
        st.session_state["ai_ctx"] = insights.build_context(include_forecasts=include_fc)
ctx = st.session_state["ai_ctx"]
with st.expander(f"Evidence pack sent to Claude (built {ctx['generated_at']}, {len(json.dumps(ctx)):,} characters)"):
    st.json(ctx, expanded=1)
    st.download_button("Download evidence pack (JSON)", json.dumps(ctx, indent=2, ensure_ascii=False).encode(),
                       "evidence_pack.json", "application/json", key="ai_ctx_dl")

if not api_key:
    st.warning("No Anthropic API key found. Add `ANTHROPIC_API_KEY=...` to the `.env` file in the project folder "
               "(then restart Streamlit) or paste a key in the sidebar.")

if st.button("Generate strategic recommendations", type="primary", disabled=not (api_key and areas), key="ai_go"):
    with st.spinner(f"Claude ({model}) is analysing the evidence… this can take a minute"):
        try:
            adv = ai.generate_advisory(ctx, areas, api_key, model, focus)
            db.save_ai_report(model, ctx, adv)
            st.session_state["ai_adv"] = adv
        except Exception as e:  # noqa: BLE001
            st.error(f"Claude API error: {type(e).__name__}: {e}")


def render(adv: dict):
    if adv.get("parse_error"):
        st.warning("Claude's response was not valid JSON; showing raw text.")
        st.markdown(adv.get("raw_text", ""))
        return
    meta = adv.get("_meta", {})
    st.caption(f"Model {meta.get('model')} · tokens in/out {meta.get('input_tokens')}/{meta.get('output_tokens')}"
               + (" · ⚠ response truncated – reduce areas" if meta.get("stop_reason") == "max_tokens" else ""))
    if adv.get("executive_summary"):
        st.markdown("### Executive summary " + ui.badge("OPINION"))
        st.info(adv["executive_summary"])
    if adv.get("key_findings"):
        st.markdown("### Key findings")
        for f in adv["key_findings"]:
            st.markdown(f"- {ui.badge(str(f.get('evidence_type', 'FACT')).upper())} {f.get('finding')} "
                        f"<span style='opacity:.6;font-size:.85em'>({f.get('source', '')})</span>", unsafe_allow_html=True)
    recs = adv.get("recommendations", [])
    if recs:
        st.markdown("### Recommendations")
        conf_rank = {"High": 0, "Medium": 1, "Low": 2}
        overview = pd.DataFrame([{"Area": r.get("area"), "Action": r.get("title"), "Confidence": r.get("confidence"),
                                  "Timeframe": r.get("implementation_timeframe")} for r in recs])
        st.dataframe(overview, hide_index=True, width="stretch")
        for r in sorted(recs, key=lambda r: conf_rank.get(r.get("confidence"), 3)):
            conf = r.get("confidence", "")
            icon = {"High": ":material/verified:", "Medium": ":material/help:", "Low": ":material/warning:"}.get(conf)
            with st.expander(f"**{r.get('area')}** — {r.get('title')}  ·  {conf} confidence", icon=icon):
                st.markdown(f"**1. Business problem**  \n{r.get('business_problem')}")
                st.markdown("**2. Supporting data**")
                for d in r.get("supporting_data", []):
                    st.markdown(f"- {ui.badge(str(d.get('evidence_type', 'FACT')).upper())} **{d.get('metric')}**: "
                                f"{d.get('value')} _(source: {d.get('source')})_")
                st.markdown(f"**3. Analytical interpretation** {ui.badge('OPINION')}  \n{r.get('analytical_interpretation')}")
                st.markdown(f"**4. Recommended action** {ui.badge('OPINION')}  \n{r.get('recommended_action')}")
                st.markdown(f"**5. Expected financial impact**  \n{r.get('expected_financial_impact')}")
                st.markdown(f"**6. Implementation timeframe**  \n{r.get('implementation_timeframe')}")
                st.markdown(f"**7. Risks & limitations**  \n{r.get('risks_and_limitations')}")
                st.markdown(f"**8. Confidence: {conf}**  \n{r.get('confidence_rationale')}")
    if adv.get("data_gaps"):
        st.markdown("### Data gaps")
        st.markdown("\n".join(f"- {g}" for g in adv["data_gaps"]))


adv = st.session_state.get("ai_adv")
hist = db.list_ai_reports()
if adv is None and not hist.empty:
    st.caption("Showing the most recent saved advisory. Generate a new one to refresh.")
    adv = json.loads(hist.iloc[0]["response_json"])
if adv:
    render(adv)
    st.divider()
    st.markdown("#### Ask a follow-up question")
    q = st.text_input("Question", key="ai_q", placeholder="e.g. What would change your pricing recommendation if Brent falls 20%?")
    if st.button("Ask Claude", disabled=not (api_key and q), key="ai_ask"):
        with st.spinner("Thinking…"):
            try:
                st.session_state["ai_ans"] = ai.ask_followup(ctx, adv, q, api_key, model)
            except Exception as e:  # noqa: BLE001
                st.error(f"Claude API error: {e}")
    if st.session_state.get("ai_ans"):
        st.markdown(st.session_state["ai_ans"])

if not hist.empty:
    with st.expander(f"Advisory history ({len(hist)} saved)"):
        st.dataframe(hist[["id", "ts", "model"]], hide_index=True, width="stretch")
        pick = st.selectbox("Load a previous advisory", hist["id"].tolist(), key="ai_hist")
        if st.button("Load", key="ai_load"):
            row = hist[hist["id"] == pick].iloc[0]
            st.session_state["ai_adv"] = json.loads(row["response_json"])
            st.rerun()
