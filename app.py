from typing import Any


import streamlit as st
from google import genai

from backend.build_base_df import build_df_base
from backend.forecast_agent import ForecastAgent
from backend.dashboard_controller import run_single_sku_dashboard, run_simulation_preview
from backend.ui_renders import (
    render_forecast,
    render_policy_tables,
    render_scenarios,
    render_optimal_scenarios,
    render_simulation_storyboard
)
from backend.simulation_agent import SimulationAgent
from backend.optimization_agent import OptimizationAgent
import plotly.graph_objects as go
import plotly.express as px
from backend.app_context import get_app_context
import pandas as pd
from datetime import datetime


HOLDOUT_START = datetime(2015, 4, 24)
HOLDOUT_END = datetime(2016, 4, 24)


st.set_page_config(
    page_title="Inventory Optimization",
    layout="wide"
)

st.title("Inventory Optimization")

# -------------------------
# LLM Client
# -------------------------

llm_client = genai.Client(api_key="YOUR_API_KEY")

# -------------------------
# Cached Data + Models
# -------------------------

@st.cache_data
def load_df():
    return build_df_base(
        store_ids=("CA_1", "TX_1", "WI_1")
    )

df_base = load_df()

@st.cache_resource
def load_forecast_model(df):
    return ForecastAgent(
        model_dir="models",
        df_base=df
    )

@st.cache_data
def load_trained_skus():
    sku_df = pd.read_csv("selected_products.csv")
    return sorted(sku_df["item_id"].unique())

trained_skus = load_trained_skus()

@st.cache_resource
def load_sim_agent():
    return SimulationAgent()

@st.cache_resource
def load_opt_agent(_sim_agent):
    return OptimizationAgent(simulation_agent=sim_agent)

forecast_agent = load_forecast_model(df_base)
sim_agent = load_sim_agent()
opt_agent = load_opt_agent(sim_agent)

# -------------------------
# Session Flags
# -------------------------

context = get_app_context()

def get_common_inputs():
    col1, col2 = st.columns(2)

    with col1:
        store_id = st.selectbox(
            "Store",
            sorted(df_base["store_id"].unique())
        )

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=HOLDOUT_START,
            min_value=HOLDOUT_START,
            max_value=HOLDOUT_END
        )

        end_date = st.date_input(
            "End Date",
            value=HOLDOUT_END,
            min_value=HOLDOUT_START,
            max_value=HOLDOUT_END
        )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    service_level_target = st.slider(
        "Service Level (%)",
        90, 99, 95
    ) / 100

    return store_id, start_date, end_date, service_level_target


def render_dashboard():
    store_id, start_date, end_date, service_level_target = get_common_inputs()

    with st.form("single_sku_form"):

        item_id = st.selectbox(
            "Product (Trained SKUs Only)",
            trained_skus
        )

        run_clicked = st.form_submit_button("Run Optimization")

    preview_clicked = st.button(
        "Preview Monte Carlo Simulation",
        key=f"preview_{item_id}"
    )

    if preview_clicked:
        baseline_context = st.session_state.get("baseline_context")
        policy = None
        assumptions = None
        if baseline_context and baseline_context.get("inputs", {}).get("item_id") == item_id:
            policies = baseline_context.get("policies", [])
            if policies:
                policy = policies[0]
            assumptions = baseline_context.get("assumptions")

        with st.spinner("Running Monte Carlo preview..."):
            preview = run_simulation_preview(
                store_id,
                item_id,
                start_date,
                end_date,
                df_base,
                forecast_agent,
                policy=policy,
                assumptions=assumptions
            )
        context["monte_carlo_preview"] = preview

    if context.get("monte_carlo_preview"):
        preview = context["monte_carlo_preview"]

        st.subheader("Monte Carlo Preview (Stochastic Sampling)")
        st.caption(
            "We simulated 500 possible demand scenarios to test this policy."
        )

        academic_tab, business_tab = st.tabs(["Academic View", "Business View"])

        with academic_tab:
            st.markdown("This view shows the raw Monte Carlo simulation outputs across runs.")
            col_a, col_b = st.columns(2)

            with col_a:
                fig_total = px.histogram(
                    preview["total_demand"],
                    nbins=30,
                    title="Total Demand Distribution"
                )
                st.plotly_chart(fig_total, use_container_width=True)

            with col_b:
                fig_fill = px.histogram(
                    [v * 100 for v in preview["fill_rate"]],
                    nbins=30,
                    title="Fill Rate Distribution (%)"
                )
                st.plotly_chart(fig_fill, use_container_width=True)

            sample_path = preview.get("sample_path", {})
            inventory_path = sample_path.get("inventory_path", [])
            stockout_flags = sample_path.get("stockout_days", [])
            reorder_days = sample_path.get("reorder_days", [])

            if inventory_path:
                fig_path = go.Figure()
                days = list(range(len(inventory_path)))

                fig_path.add_trace(go.Scatter(
                    x=days,
                    y=inventory_path,
                    mode="lines",
                    name="Inventory Level",
                    line=dict(width=3)
                ))

                stockout_x = [days[i] for i, flag in enumerate(stockout_flags) if flag]
                stockout_y = [inventory_path[i] for i, flag in enumerate(stockout_flags) if flag]
                if stockout_x:
                    fig_path.add_trace(go.Scatter(
                        x=stockout_x,
                        y=stockout_y,
                        mode="markers",
                        name="Stockout Day",
                        marker=dict(color="red", size=8)
                    ))

                reorder_x = [days[i] for i in reorder_days if i < len(days)]
                reorder_y = [inventory_path[i] for i in reorder_days if i < len(inventory_path)]
                if reorder_x:
                    fig_path.add_trace(go.Scatter(
                        x=reorder_x,
                        y=reorder_y,
                        mode="markers",
                        name="Reorder Placed",
                        marker=dict(color="orange", size=8, symbol="triangle-up")
                    ))

                fig_path.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Day",
                    yaxis_title="Inventory Level",
                    title="Inventory Timeline (Single Simulation Run)"
                )
                st.plotly_chart(fig_path, use_container_width=True)

        with business_tab:
            mean_fill_rate = preview.get("mean_fill_rate", 0.0)
            stockout_prob = preview.get("stockout_probability", 0.0)
            worst_lost_units = preview.get("worst_case_lost_units", 0.0)

            metric_cols = st.columns(3)
            metric_cols[0].metric("Average Service Level", f"{mean_fill_rate:.2%}")
            metric_cols[1].metric("Probability of Stockout", f"{stockout_prob:.2%}")
            metric_cols[2].metric("Worst Case Lost Units", f"{worst_lost_units:.2f}")

            st.markdown(
                f"This policy meets demand about {mean_fill_rate * 100:.1f}% of the time "
                f"with a {stockout_prob * 100:.1f}% chance of stockout."
            )

    if run_clicked:

        with st.spinner("Running optimization..."):

            results = run_single_sku_dashboard(
                store_id,
                item_id,
                start_date,
                end_date,
                service_level_target,
                "auto",
                None,
                None,
                df_base,
                forecast_agent,
                llm_client
            )

            context["single_sku_results"] = results
            st.session_state["baseline_context"] = results

    # -------------------------
    # Render Results (Only if Flag True)
    # -------------------------

    if context.get("single_sku_results"):
        results = context["single_sku_results"]

        # Use containers to prevent full redraw flicker
        forecast_container = st.container()
        storyboard_container = st.container()
        policy_container = st.container()
        scenario_container = st.container()
        optimal_container = st.container()

        with forecast_container:
            render_forecast(results)

        with storyboard_container:
            render_simulation_storyboard(results)

        with policy_container:
            render_policy_tables(results)

        with scenario_container:
            render_scenarios(results)

        with optimal_container:
            render_optimal_scenarios(results)

render_dashboard()
