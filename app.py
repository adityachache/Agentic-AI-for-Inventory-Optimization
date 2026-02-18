import os
import streamlit as st
import pandas as pd
from google import genai
from backend.agent_pipeline import run_inventory_planner
from backend.build_base_df import build_df_base
from backend.forecast_agent import ForecastAgent

# Use env var so the key is never committed (set GOOGLE_GENAI_API_KEY locally or in .env)
llm_client = genai.Client(api_key=os.environ.get("GOOGLE_GENAI_API_KEY", ""))

@st.cache_data
def load_df_base():
    return build_df_base(
        store_ids=("CA_1", "TX_1", "WI_1"),
        max_ids=3000
    )

df_base = load_df_base()

agent = ForecastAgent(
    model_dir="models",
    df_base=df_base
)

print(df_base.shape)

st.header("Inventory Optimization")

st.markdown(
    """
    Select products and planning preferences below to generate
    demand forecasts and optimized inventory replenishment policies.
    """
)

if "planner_results" not in st.session_state:
    st.session_state.planner_results = None

if "last_run_inputs" not in st.session_state:
    st.session_state.last_run_inputs = None


# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("Planning Inputs")

col1, col2 = st.columns(2)

with col1:
    store_id = st.selectbox(
        "Store",
        options=sorted(df_base["store_id"].unique()),
        index=0
    )

    item_ids = st.multiselect(
        "Products (max 5)",
        options=sorted(df_base["item_id"].unique()),
        max_selections=5,
        help="Select up to 5 products for focused planning."
    )

with col2:
    start_date = st.date_input(
        "Start date",
        value=df_base["date"].min()
    )
    end_date = st.date_input(
        "End date",
        value=df_base["date"].max()
    )

service_level_target = st.slider(
    "Target Service Level (%)",
    min_value=90,
    max_value=99,
    value=95,
    step=1
) / 100.0

mode = st.radio(
    "Policy Mode",
    options=["Auto Optimize", "Manual What-If"],
    horizontal=True
)

manual_policy = None
if mode == "Manual What-If":
    c1, c2 = st.columns(2)
    with c1:
        s_val = st.number_input("Reorder Point (s)", min_value=0, value=100)
    with c2:
        q_val = st.number_input("Order Quantity (Q)", min_value=1, value=200)

    manual_policy = {"s": int(s_val), "Q": int(q_val)}

# -------------------------
# ADVANCED ASSUMPTIONS
# -------------------------
with st.expander("Advanced Planning Assumptions"):
    lead_time_days = st.number_input("Lead Time (days)", value=2)
    order_cost = st.number_input("Order Cost ($ per order)", value=20.0)
    holding_cost_rate = st.number_input("Holding Cost Rate (annual %)", value=25.0) / 100
    stockout_cost_multiplier = st.number_input("Stockout Cost Multiplier", value=3.0)
    demand_cv = st.number_input("Demand Variability (CV)", value=0.25)

assumptions = {
    "lead_time_days": lead_time_days,
    "order_cost": order_cost,
    "holding_cost_rate": holding_cost_rate,
    "stockout_cost_multiplier": stockout_cost_multiplier,
    "demand_cv": demand_cv
}

# -------------------------
# RUN BUTTON
# -------------------------
run_clicked = st.button("Run Inventory Optimization", type="primary")

# -------------------------
# RESULTS SECTION
# -------------------------
# --- COMPUTE ONLY WHEN BUTTON CLICKED ---
if run_clicked:
    if not item_ids:
        st.error("Please select at least one product.")
    else:
        with st.spinner("Running demand forecasting, simulation, and optimization..."):
            results = run_inventory_planner(
                store_id=store_id,
                item_ids=item_ids,
                start_date=start_date,
                end_date=end_date,
                service_level_target=service_level_target,
                mode="auto" if mode == "Auto Optimize" else "manual",
                manual_policy=manual_policy,
                assumptions=assumptions,
                df_base=df_base,
                forecast_agent=agent,
                llm_client=llm_client
            )

        st.session_state.planner_results = results
        st.session_state.last_run_inputs = {
            "store_id": store_id,
            "item_ids": item_ids,
            "start_date": start_date,
            "end_date": end_date,
            "service_level_target": service_level_target,
            "mode": mode
        }

# --- DISPLAY RESULTS IF THEY EXIST ---
if st.session_state.planner_results is not None:
    results = st.session_state.planner_results
    st.success("Analysis complete")

    # (all your forecast/policy/cost/LLM rendering here)    
    # render forecast, policy, cost, summaries

    # -------------------------
    # DEMAND FORECASTS
    # -------------------------
    st.subheader("Demand Forecasts")

    for item_id in item_ids:
        st.markdown(f"**{item_id}**")
        forecast_df = results["forecast"][item_id]["daily_forecast"]
        forecast_df = forecast_df.set_index("date")

        st.line_chart(forecast_df["forecast"])

        st.caption(
            f"Total forecast: {int(results['forecast'][item_id]['total_units'])} units "
            f"({results['forecast'][item_id]['avg_daily_units']:.1f} units/day)"
        )

    # -------------------------
    # POLICY TABLE
    # -------------------------
    st.subheader("Recommended Inventory Policies")

    policy_df = pd.DataFrame(results["policies"])
    st.dataframe(
        policy_df[
            ["item_id", "s", "Q", "fill_rate", "total_cost"]
        ],
        use_container_width=True
    )

    # -------------------------
    # COST BREAKDOWN
    # -------------------------
    st.subheader("Cost Breakdown")

    st.dataframe(
        policy_df[
            ["item_id", "holding_cost", "ordering_cost", "stockout_cost"]
        ],
        use_container_width=True
    )

    # -------------------------
    # MANAGERIAL SUMMARIES
    # -------------------------
    st.subheader("Managerial Recommendations")

    for item_id, summary in results["llm_summaries"].items():
        with st.expander(f"{item_id} – Recommendation"):
            st.write(summary)