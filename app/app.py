"""Streamlit demo: trained logistic regression vs. zero-shot Claude Haiku on telecom churn.

Reads only ``app/data/`` (see ``export_app_data.py``) and ``data/processed/``.
It never reads ``data/raw`` and never calls an API.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as d  # noqa: E402
import predictor  # noqa: E402

logger = logging.getLogger(__name__)

BASELINE = "Logistic regression"
HAIKU = "Claude Haiku"
# Blue is always the trained baseline; orange is always Claude Haiku.
COLORS = {BASELINE: "#2A78D6", HAIKU: "#EB6834"}
NEUTRAL = "#B0B0B0"
STAY_GRAY = "#8E959E"
REPO_URL = "https://github.com/Denver-A7/churn-llm-comparison"

st.set_page_config(page_title="Churn: trained model vs. zero-shot LLM", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data
def get_tables():
    return d.load_customer_table(), d.load_segments(), d.load_metrics(), d.load_model_spec()


table, segments, metrics, model_spec = get_tables()
disagreements = table[~table["agree"]]

n = metrics["n_compared"]
n_agree = metrics["n_agree"]
n_dis = metrics["n_disagree"]
haiku_wins = metrics["haiku_right_on_disagreements"]
base_wins = metrics["baseline_right_on_disagreements"]
p_value = metrics["sign_test_p_value"]

m_base = metrics["baseline_compared"]
m_llm = metrics["haiku_compared"]
m_full = metrics["baseline_full_test"]


def pct(x: float) -> str:
    return f"{x:.1%}"


def render_tab(tab, render: Callable[[], None]) -> None:
    """Draw one tab; on failure show a short message instead of a traceback."""
    with tab:
        try:
            render()
        except Exception:
            logger.exception("Tab renderer %s failed", render.__name__)
            st.error(
                "This section could not be displayed. Try reloading the page. If it keeps happening, "
                "regenerate the demo data with `python app/export_app_data.py`."
            )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### About this demo")
    st.markdown(
        f"A trained logistic regression and Claude Haiku (zero-shot) each predicted churn for the same "
        f"{n} telecom customers. This app reads the saved results; it never calls an API."
    )
    st.markdown(
        f"<span style='color:{COLORS[BASELINE]}'>&#9632;</span> Logistic regression &nbsp; "
        f"<span style='color:{COLORS[HAIKU]}'>&#9632;</span> Claude Haiku",
        unsafe_allow_html=True,
    )
    st.markdown("### Data")
    st.markdown(
        "IBM Telco Customer Churn sample dataset, as published on "
        "[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). "
        "It contains no real customers."
    )
    st.markdown(f"[Code and write-up on GitHub]({REPO_URL})")
    st.markdown("### Limits")
    st.markdown(
        f"""
- Only {n} customers ({metrics['n_churners_compared']} churners), one run.
- One AI model and one prompt; other models or prompts could behave differently.
- The AI's reasons are generated text, not a record of how it decided.
- p = {p_value:.2f} on the disagreements means the gap between the models could be chance.
        """
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Can a zero-shot LLM predict churn as well as a trained model?")
st.markdown(
    f"A trained **logistic regression** and **Claude Haiku** (given no training, only a short "
    f"plain-language description of each customer) each predicted whether **{n} telecom customers** "
    f"would leave. They scored about the same, but made different mistakes. "
    f"This demo shows where they agreed, where they differed, and whether Haiku's written reasons hold up."
)

tab_overview, tab_metrics, tab_dis, tab_seg, tab_reason, tab_whatif = st.tabs(
    ["Overview", "Metrics", "Disagreements", "Segments", "Reasoning", "What-if"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Customers compared", n,
        help=f"Haiku answered {n} of {metrics['n_sampled']} sampled test customers in the required format.",
    )
    c2.metric("Models agreed", f"{n_agree} ({pct(n_agree / n)})")
    c3.metric("Haiku right when they disagreed", f"{haiku_wins} of {n_dis}")
    c4.metric("Baseline right when they disagreed", f"{base_wins} of {n_dis}")

    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(
            go.Pie(
                labels=["Agreed", "Disagreed"],
                values=[n_agree, n_dis],
                hole=0.55,
                marker_colors=[NEUTRAL, "#555555"],
                textinfo="label+value",
                sort=False,
            )
        )
        fig.update_layout(title="Did the two models give the same answer?", showlegend=False, height=380)
        st.plotly_chart(fig)
    with right:
        fig = go.Figure(
            go.Bar(
                x=[HAIKU, BASELINE],
                y=[haiku_wins, base_wins],
                marker_color=[COLORS[HAIKU], COLORS[BASELINE]],
                text=[haiku_wins, base_wins],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Who was right on the {n_dis} customers where they disagreed?",
            yaxis_title="Customers",
            height=380,
            yaxis_range=[0, max(haiku_wins, base_wins) * 1.25],
        )
        st.plotly_chart(fig)

    st.markdown(
        f"Haiku was right {haiku_wins} times and the baseline {base_wins}. A sign test gives "
        f"**p ≈ {p_value:.2f}**, so this is not a meaningful difference: with only {n_dis} disagreements, "
        f"a split like this could easily happen by chance."
    )
    with st.expander("What this comparison is and is not"):
        st.markdown(
            """
- Both models saw the **same customers** from the same held-out test set. Haiku was never shown the answers, customer IDs, or the baseline's predictions.
- The baseline is a standard logistic regression trained on 80% of the data. Haiku received a one-paragraph description of each customer and was asked for *Yes/No* plus one short reason.
- "Right" means the prediction matched whether the customer actually left.
- Haiku's reasons are text to **inspect**, not proof of how it decided. The *Reasoning* tab looks at them.
- This is one run, one prompt, one LLM, on 194 customers (54 of whom churned). Treat the numbers as a pilot, not a verdict.
            """
        )

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def render_metrics() -> None:
    st.subheader(f"Scores on the same {n} customers")
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
    long = pd.DataFrame(
        [{"Measure": labels[k], "Model": BASELINE, "Score": m_base[k]} for k in metric_names]
        + [{"Measure": labels[k], "Model": HAIKU, "Score": m_llm[k]} for k in metric_names]
    )
    fig = px.bar(
        long, x="Measure", y="Score", color="Model", barmode="group",
        color_discrete_map=COLORS, text=long["Score"].map(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], height=420, legend_title=None)
    st.plotly_chart(fig)

    with st.expander("What these measures mean"):
        st.markdown(
            """
- **Accuracy** – share of all customers the model got right.
- **Precision** – of the customers the model said would leave, how many really did. Low precision means many false alarms.
- **Recall** – of the customers who really left, how many the model caught. Low recall means many missed churners.
- **F1** – a single number balancing precision and recall.
            """
        )

    st.subheader("Where the mistakes were")
    cm_left, cm_right = st.columns(2)
    for col, name, m in [(cm_left, BASELINE, m_base), (cm_right, HAIKU, m_llm)]:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig = px.imshow(
            z,
            x=["Predicted: stays", "Predicted: leaves"],
            y=["Actually stayed", "Actually left"],
            text_auto=True,
            color_continuous_scale=[[0, "#f5f5f5"], [1, COLORS[name]]],
        )
        fig.update_layout(title=name, coloraxis_showscale=False, height=360)
        fig.update_traces(textfont_size=20)
        col.plotly_chart(fig)
    st.markdown(
        f"The baseline **missed {m_base['fn']} churners** and raised **{m_base['fp']} false alarms**; "
        f"Haiku missed **{m_llm['fn']}** and raised **{m_llm['fp']}**. Haiku's slightly better scores come "
        f"mostly from fewer false alarms, at the cost of one more missed churner."
    )

    st.subheader("Cost-weighted view")
    st.markdown(
        "Missing a churner is usually more expensive than a false alarm. If a miss costs **5×** a false alarm, "
        "the normalized scores (1 = perfect, 0 = worst possible) are:"
    )
    cw1, cw2 = st.columns(2)
    cw1.metric(BASELINE, f"{m_base['cost_weighted_score']:.3f}")
    cw2.metric(HAIKU, f"{m_llm['cost_weighted_score']:.3f}")
    st.caption("The 5:1 weight is an assumption, not a measured business cost. The two scores are effectively tied.")

    with st.expander("Baseline on the full test set"):
        st.markdown(
            f"On all **{m_full['n']:,}** held-out customers (not just the {n} Haiku answered), the "
            f"baseline scored accuracy {m_full['accuracy']:.3f}, precision {m_full['precision']:.3f}, "
            f"recall {m_full['recall']:.3f}, F1 {m_full['f1']:.3f}. The 194-customer table above is the fair "
            f"comparison because both models answered exactly those customers."
        )

# --------------------------------------------------------------------------- #
# Disagreements
# --------------------------------------------------------------------------- #
def render_disagreements() -> None:
    st.subheader("Browse the customers")
    f1, f2, f3, f4 = st.columns([1, 1.3, 1.3, 1.3])
    only_dis = f1.toggle(f"Only the {n_dis} disagreements", value=True)
    who_options = ["Haiku right", "Baseline right", "Both right", "Both wrong"]
    who_sel = f2.multiselect("Who was right", who_options, default=who_options)
    no_internet_only = f3.checkbox("Haiku's reason mentions 'no internet service'")
    contract_sel = f4.multiselect(
        "Contract", ["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"],
    )
    tenure_sel = st.multiselect("Tenure", d.TENURE_LABELS, default=d.TENURE_LABELS)

    view = disagreements if only_dis else table
    view = view[view["who_was_right"].isin(who_sel)]
    if no_internet_only:
        view = view[view["mentions_no_internet"]]
    view = view[view["Contract"].isin(contract_sel) & view["tenure_bucket"].isin(tenure_sel)]

    show_cols = [
        "customerID", "actual_churn", "baseline_predicted", "baseline_probability", "llm_predicted",
        "who_was_right", "Contract", "tenure", "MonthlyCharges", "llm_reason",
    ]
    st.caption(f"{len(view)} customers match.")
    st.dataframe(
        view[show_cols],
        width="stretch",
        hide_index=True,
        height=min(420, 38 + 35 * max(len(view), 1)),
        column_config={
            "customerID": "Customer",
            "actual_churn": "Actually left?",
            "baseline_predicted": "Baseline says",
            "baseline_probability": st.column_config.ProgressColumn(
                "Baseline churn probability", min_value=0, max_value=1, format="%.2f"
            ),
            "llm_predicted": "Haiku says",
            "who_was_right": "Who was right",
            "Contract": "Contract",
            "tenure": "Tenure (months)",
            "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "llm_reason": st.column_config.TextColumn("Haiku's reason", width="large"),
        },
    )

    st.subheader("Look at one customer")
    if view.empty:
        st.warning("No customers match the current filters.")
    else:
        picked = st.selectbox("Customer", view["customerID"].tolist())
        row = table.set_index("customerID").loc[picked]
        o1, o2, o3 = st.columns(3)
        o1.metric("Actually left?", row["actual_churn"])
        o2.metric("Baseline said", row["baseline_predicted"], help=f"Churn probability {row['baseline_probability']:.2f}")
        o3.metric("Haiku said", row["llm_predicted"])
        verdict = {
            "Both right": "Both models got this one right.",
            "Both wrong": "Both models got this one wrong.",
            "Haiku right": "Haiku was right; the baseline was wrong.",
            "Baseline right": "The baseline was right; Haiku was wrong.",
        }[row["who_was_right"]]
        st.markdown(f"**{verdict}**")
        st.markdown(f"> *Haiku's reason:* {row['llm_reason']}")
        st.markdown("**Customer profile** (what both models were told)")
        profile = {c: row[c] for c in d.PROFILE_COLUMNS}
        profile["SeniorCitizen"] = "Yes" if profile["SeniorCitizen"] == 1 else "No"
        profile["MonthlyCharges"] = f"${profile['MonthlyCharges']:.2f}"
        total = profile["TotalCharges"]
        profile["TotalCharges"] = "n/a" if pd.isna(total) else f"${total:,.2f}"
        profile["tenure"] = f"{int(profile['tenure'])} months"
        items = list(profile.items())
        half = (len(items) + 1) // 2
        p1, p2 = st.columns(2)
        p1.table(pd.DataFrame(items[:half], columns=["Field", "Value"]).set_index("Field"))
        p2.table(pd.DataFrame(items[half:], columns=["Field", "Value"]).set_index("Field"))

# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
def render_segments() -> None:
    st.subheader("Where did the models disagree, and who was right there?")
    dim_labels = {"Contract": "Contract type", "tenure_bucket": "Tenure", "cost_quartile": "Monthly charge quartile"}
    dim = st.radio("Break down by", list(dim_labels), format_func=dim_labels.get, horizontal=True)
    seg = segments[segments["segment_dimension"] == dim].copy()
    seg["label"] = seg["segment"] + "  (n=" + seg["rows_with_valid_predictions"].astype(str) + ")"

    s1, s2 = st.columns(2)
    with s1:
        fig = px.bar(
            seg, x="label", y="disagreement_rate",
            text=seg["disagreement_rate"].map(pct),
            color_discrete_sequence=["#555555"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            title="How often the models disagreed", xaxis_title=None,
            yaxis_title="Share of customers", yaxis_tickformat=".0%", yaxis_range=[0, 0.4], height=400,
        )
        st.plotly_chart(fig)
    with s2:
        stacked = pd.concat(
            [
                pd.DataFrame({"label": seg["label"], "Model": BASELINE, "Correct": seg["baseline_correct_on_disagreements"]}),
                pd.DataFrame({"label": seg["label"], "Model": HAIKU, "Correct": seg["llm_correct_on_disagreements"]}),
            ]
        )
        fig = px.bar(stacked, x="label", y="Correct", color="Model", color_discrete_map=COLORS, text="Correct")
        fig.update_layout(
            title="On the disagreements, who was right", xaxis_title=None,
            yaxis_title="Customers", height=400, legend_title=None,
        )
        st.plotly_chart(fig)

    if dim == "Contract":
        st.markdown(
            "**30 of the 34 disagreements were month-to-month customers.** On two-year contracts the models never disagreed: "
            "those customers almost all stay, and both models know it."
        )
    elif dim == "tenure_bucket":
        st.markdown(
            "Haiku lost most of the disagreements among **new customers (0–12 months)** but won most among "
            "**longer-tenure customers**. It behaves a lot like a tenure rule, which the chart below shows directly."
        )
    else:
        st.markdown("Disagreements cluster in the **top charge quartile**, where Haiku was right more often.")

    st.subheader("Haiku behaves a lot like a tenure rule")
    g = table.groupby("tenure_bucket")
    tenure_df = pd.DataFrame(
        {
            "Tenure": d.TENURE_LABELS,
            "Actually left": [g.get_group(b)["actual_churn"].eq("Yes").mean() for b in d.TENURE_LABELS],
            BASELINE: [g.get_group(b)["baseline_predicted"].eq("Yes").mean() for b in d.TENURE_LABELS],
            HAIKU: [g.get_group(b)["llm_predicted"].eq("Yes").mean() for b in d.TENURE_LABELS],
        }
    ).melt(id_vars="Tenure", var_name="Series", value_name="Share predicted/actually churned")
    fig = px.bar(
        tenure_df, x="Tenure", y="Share predicted/actually churned", color="Series", barmode="group",
        color_discrete_map={**COLORS, "Actually left": NEUTRAL},
        text=tenure_df["Share predicted/actually churned"].map(lambda v: f"{v:.0%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1], height=420, legend_title=None, xaxis_title=None)
    st.plotly_chart(fig)

    acc = pd.DataFrame(
        {
            "Tenure": d.TENURE_LABELS,
            "Customers": [len(g.get_group(b)) for b in d.TENURE_LABELS],
            f"{BASELINE} accuracy": [
                (g.get_group(b)["baseline_predicted"] == g.get_group(b)["actual_churn"]).mean() for b in d.TENURE_LABELS
            ],
            f"{HAIKU} accuracy": [
                (g.get_group(b)["llm_predicted"] == g.get_group(b)["actual_churn"]).mean() for b in d.TENURE_LABELS
            ],
        }
    )
    st.dataframe(
        acc, hide_index=True, width="stretch",
        column_config={
            f"{BASELINE} accuracy": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            f"{HAIKU} accuracy": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        },
    )
    st.markdown(
        "Haiku predicts churn for almost every customer in their first year and almost nobody after that. "
        "That makes it more accurate than the baseline on long-tenure customers and less accurate on new ones."
    )

# --------------------------------------------------------------------------- #
# Reasoning
# --------------------------------------------------------------------------- #
def render_reasoning() -> None:
    st.subheader("Do Haiku's written reasons hold up?")
    st.markdown(
        "Each Haiku prediction came with one sentence of justification. The facts it quotes are taken from the "
        "customer description, so they are usually true. The question is whether the **logic** is sound. "
        "A reason is plausible text, not proof of what drove the prediction, so it needs auditing."
    )

    ni = table[table["mentions_no_internet"]]
    ni_stayed = int((ni["actual_churn"] == "No").sum())
    r1, r2, r3 = st.columns(3)
    r1.metric("Reasons citing 'no internet service' as a risk", len(ni))
    r2.metric("…of those customers who actually stayed", f"{ni_stayed} of {len(ni)}")
    r3.metric(
        "Churn rate of no-internet customers in the full dataset",
        pct(metrics["no_internet_reasons"]["dataset_churn_rate_no_internet"]),
    )
    st.markdown(
        "Haiku repeatedly treated having **no internet service** as a warning sign, always alongside a month-to-month "
        "contract and short tenure. In this dataset, customers without internet are among the *least* likely to leave, "
        "and the baseline's coefficients do not treat it as a churn driver. That is a pattern worth checking before trusting the reasons."
    )

    st.subheader("Read the reasons")
    q1, q2, q3 = st.columns([1.2, 1.2, 2])
    outcome = q1.selectbox(
        "Show", ["All customers", "Haiku was right", "Haiku was wrong", "Only disagreements"],
    )
    pattern_only = q2.checkbox("Only reasons citing 'no internet service'")
    search = q3.text_input("Search the reasons", placeholder="e.g. tenure, electronic check, tech support")

    rv = table
    if outcome == "Haiku was right":
        rv = rv[rv["llm_predicted"] == rv["actual_churn"]]
    elif outcome == "Haiku was wrong":
        rv = rv[rv["llm_predicted"] != rv["actual_churn"]]
    elif outcome == "Only disagreements":
        rv = rv[~rv["agree"]]
    if pattern_only:
        rv = rv[rv["mentions_no_internet"]]
    if search.strip():
        rv = rv[rv["llm_reason"].fillna("").str.contains(search.strip(), case=False, regex=False)]

    st.caption(f"{len(rv)} reasons match.")
    reason_cols = ["customerID", "actual_churn", "llm_predicted", "who_was_right", "Contract", "tenure", "InternetService", "llm_reason"]
    st.dataframe(
        rv[reason_cols], hide_index=True, width="stretch", height=500,
        column_config={
            "customerID": "Customer",
            "actual_churn": "Actually left?",
            "llm_predicted": "Haiku said",
            "who_was_right": "Who was right",
            "tenure": "Tenure (months)",
            "InternetService": "Internet",
            "llm_reason": st.column_config.TextColumn("Haiku's reason", width="large"),
        },
    )
    with st.expander("What a close read of five examples showed"):
        st.markdown(
            """
Five reasons were checked by hand against the customers' full records, picking cases where the two models were right or wrong in different combinations:

- In all five, the **facts Haiku quoted were true**.
- The **logic fully held up in one** (long tenure, tech support, automatic payment).
- Two treated **"no internet service" as a risk**, although those customers rarely leave.
- Two let **long tenure outweigh several real risk factors** (fiber optic, electronic-check payment, high monthly charges).
- One prediction was **right for a flawed reason** – a correct answer does not mean a sound explanation.

Five hand-picked cases illustrate patterns; they do not represent all 194 customers.
            """
        )

# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
def render_whatif() -> None:
    st.subheader("What would the trained model say about a customer like this?")
    st.markdown(
        "This scores a customer you describe with the **same logistic regression** used in the study, using its "
        "exported coefficients and preprocessing (no retraining, no scikit-learn at runtime). "
        "Haiku is not involved: the app never calls an API."
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
        auto_total = st.checkbox("Estimate total charges as tenure × monthly", value=True)
        total = (
            tenure * monthly
            if auto_total
            else st.number_input("Total charges ($)", min_value=0.0, value=float(tenure * monthly), step=10.0)
        )
    with b:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addons = {}
        for label, key in [
            ("Online security", "OnlineSecurity"),
            ("Online backup", "OnlineBackup"),
            ("Device protection", "DeviceProtection"),
            ("Tech support", "TechSupport"),
            ("Streaming TV", "StreamingTV"),
            ("Streaming movies", "StreamingMovies"),
        ]:
            addons[key] = (
                st.selectbox(label, ["No", "Yes"], key=f"whatif_{key}")
                if internet != "No"
                else "No internet service"
            )
        if internet == "No":
            st.caption("Add-ons are set to 'No internet service'.")
    with c:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    prob = predictor.predict_proba(model_spec, customer)
    threshold = model_spec["threshold"]
    label = "Likely to leave" if prob >= threshold else "Likely to stay"

    st.divider()
    g1, g2 = st.columns([1, 2])
    g1.metric("Baseline prediction", label, help=f"Threshold {threshold}, the same the study used.")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Churn probability (logistic regression)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS[BASELINE]},
                "steps": [{"range": [0, 50], "color": "#f0f0f0"}, {"range": [50, 100], "color": "#e0e0e0"}],
                "threshold": {"line": {"color": "#333", "width": 3}, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=280, margin={"t": 60, "b": 10})
    g2.plotly_chart(fig)
    st.caption(
        "The model was trained with balanced class weights, so it leans toward flagging churn; expect many "
        "month-to-month customers to land above 50%. This is a demo of the trained baseline, not a production scoring tool."
    )

    st.subheader("What is pushing this customer's score")
    contrib = pd.DataFrame(predictor.contributions(model_spec, customer))
    contrib = contrib[contrib["contribution"].abs() > 1e-9]
    contrib["size"] = contrib["contribution"].abs()
    contrib = contrib.sort_values("size", ascending=True)  # Plotly draws the last row at the top
    contrib["direction"] = contrib["contribution"].map(lambda v: "Toward leaving" if v > 0 else "Toward staying")
    fig = go.Figure(
        go.Bar(
            x=contrib["contribution"],
            y=contrib["label"],
            orientation="h",
            marker_color=[COLORS[BASELINE] if v > 0 else STAY_GRAY for v in contrib["contribution"]],
            text=contrib["contribution"].map(lambda v: f"{v:+.2f}"),
            textposition="outside",
            cliponaxis=False,
            customdata=contrib["direction"],
            hovertemplate="%{y}<br>%{x:+.3f} (%{customdata})<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)
    fig.update_layout(
        height=max(360, 26 * len(contrib) + 80),
        xaxis_title="Contribution to the score (coefficient × this customer's value)",
        yaxis_title=None,
        margin={"l": 10, "r": 60, "t": 20, "b": 50},
    )
    st.plotly_chart(fig)
    st.caption(
        f"Blue bars push toward *leaves*, gray bars toward *stays*. Numeric features are standardized first, so the "
        f"bar reflects how unusual this customer's value is. The model's intercept ({model_spec['intercept']:+.2f}) "
        f"is not shown."
    )

    with st.expander("All coefficients"):
        st.markdown(
            "Coefficients on standardized numeric features and one-hot categories. Positive pushes toward "
            "*leaves*, negative toward *stays*."
        )
        rows = [
            {"Feature": name, "Coefficient": col["coef"]} for name, col in model_spec["numeric"].items()
        ] + [
            {"Feature": f"{name} = {cat}", "Coefficient": coef}
            for name, col in model_spec["categorical"].items()
            for cat, coef in col["coefs"].items()
        ]
        coef_df = pd.DataFrame(rows).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(
            coef_df, hide_index=True, width="stretch", height=380,
            column_config={"Coefficient": st.column_config.NumberColumn(format="%.3f")},
        )


# --------------------------------------------------------------------------- #
# Draw the tabs
# --------------------------------------------------------------------------- #
render_tab(tab_overview, render_overview)
render_tab(tab_metrics, render_metrics)
render_tab(tab_dis, render_disagreements)
render_tab(tab_seg, render_segments)
render_tab(tab_reason, render_reasoning)
render_tab(tab_whatif, render_whatif)

