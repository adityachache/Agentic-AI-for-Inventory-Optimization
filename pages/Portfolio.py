import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from backend.build_base_df import build_df_base
from backend.forecast_agent import ForecastAgent
from backend.simulation_agent import SimulationAgent
from backend.optimization_agent import OptimizationAgent
from backend.portfolio_controller import generate_portfolio_candidates
from backend.portfolio_optimizer import solve_portfolio
from backend.app_context import get_app_context


HOLDOUT_START = datetime(2015, 4, 24)
HOLDOUT_END = datetime(2016, 4, 24)

st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide"
)

st.title("Portfolio Optimizer")

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

context = get_app_context()

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

with st.form("portfolio_form"):
    sku_list = st.multiselect(
        "Select up to 10 SKUs (Trained Set)",
        trained_skus,
        max_selections=10
    )

    budget = st.number_input(
        "Inventory Capital Budget ($)",
        value=5000
    )

    run_portfolio = st.form_submit_button("Run Portfolio Optimization")

if run_portfolio:
    if len(sku_list) == 0:
        st.error("Select at least one SKU.")
    else:
        with st.spinner("Generating candidate policies..."):
            candidate_data = generate_portfolio_candidates(
                sku_list,
                store_id,
                start_date,
                end_date,
                service_level_target,
                df_base,
                forecast_agent,
                sim_agent,
                opt_agent
            )

        with st.spinner("Solving portfolio optimization..."):
            selected = solve_portfolio(candidate_data, budget)

            if "error" in selected:
                st.error(selected["error"])
                st.stop()

        context["portfolio_results"] = selected
        context["portfolio_meta"] = {
            "store_id": store_id,
            "start_date": start_date,
            "end_date": end_date,
            "service_level_target": service_level_target,
            "budget": budget
        }

if context.get("portfolio_results"):
    selected = context["portfolio_results"]
    portfolio_meta = context.get("portfolio_meta", {})
    if "budget" in portfolio_meta:
        budget = portfolio_meta["budget"]

    rows = []
    total_mgmt_cost = 0
    total_capital = 0
    total_procurement = 0

    for sku, policy in selected.items():
        avg_price = (
            df_base[
                (df_base["item_id"] == sku) &
                (df_base["store_id"] == store_id)
            ]["price"]
            .mean()
        )
        fixed_order_cost = opt_agent.order_cost
        cost_per_order = (avg_price * policy["Q"]) + fixed_order_cost

        rows.append({
            "SKU": sku,
            "Reorder Point (units)": policy["s"],
            "Order Quantity (units)": policy["Q"],
            "Cost per Order ($)": round(cost_per_order, 2),
            "Total Inventory Operating Cost ($)": round(policy["cost"], 2),
            "Working Capital Tied Up ($)": round(policy["investment"], 2),
            "Annual Purchase Volume ($)": round(policy["procurement_spend"], 2)
        })

        total_mgmt_cost += policy["cost"]
        total_capital += policy["investment"]
        total_procurement += policy["procurement_spend"]

    portfolio_df = pd.DataFrame(rows)

    with st.container():
        st.subheader("Selected Portfolio Policies")
        portfolio_df = portfolio_df[
            [
                "SKU",
                "Reorder Point (units)",
                "Order Quantity (units)",
                "Cost per Order ($)",
                "Total Inventory Operating Cost ($)",
                "Working Capital Tied Up ($)",
                "Annual Purchase Volume ($)"
            ]
        ]
        st.dataframe(portfolio_df, use_container_width=True)

    with st.container():
        st.markdown("## Portfolio Financial Summary")

        capital_utilization = (total_capital / budget) * 100 if budget > 0 else 0
        remaining_capital = budget - total_capital

        planning_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        annualized_cost = total_mgmt_cost * (365 / planning_days) if planning_days > 0 else total_mgmt_cost

        metric_row_1 = st.columns(4)
        metric_row_1[0].metric("💰 Budget", f"${budget:,.2f}")
        metric_row_1[1].metric("📦 Working Capital", f"${total_capital:,.2f}")
        metric_row_1[2].metric("📉 Operating Cost", f"${total_mgmt_cost:,.2f}")
        metric_row_1[3].metric("🛒 Annual Purchase Volume", f"${total_procurement:,.2f}")

        metric_row_2 = st.columns(3)
        metric_row_2[0].metric("Remaining Capital", f"${remaining_capital:,.2f}")
        metric_row_2[1].metric("Capital Utilization", f"{capital_utilization:.2f}%")
        metric_row_2[2].metric("Annualized Operating Cost", f"${annualized_cost:,.2f}")

        st.caption(
            f"This solution allocates {capital_utilization:.1f}% of available capital "
            f"while minimizing operating cost."
        )
        st.caption(
            "If additional capital were allocated, total operating cost could reduce further by 0%."
        )

    with st.container():
        st.markdown("## Portfolio Risk Overview")

        demand_totals = {}
        sim_risk = {}
        total_demand_all = 0.0

        for sku in selected.keys():
            forecast_result = forecast_agent.forecast(
                item_id=sku,
                store_id=store_id,
                start_date=start_date,
                end_date=end_date
            )
            forecast_df = forecast_result["results"][0]["daily_forecast"]
            sku_demand = float(forecast_df["forecast"].sum())
            demand_totals[sku] = sku_demand
            total_demand_all += sku_demand

            sim_eval = sim_agent.simulate(
                forecast_df=forecast_df,
                item_id=sku,
                store_id=store_id,
                s=selected[sku]["s"],
                Q=selected[sku]["Q"],
                return_sample_path=False,
                return_full_distribution=False
            )
            sim_risk[sku] = {
                "fill_rate": sim_eval["results"]["expected_fill_rate"],
                "stockout_probability": sim_eval["results"]["stockout_probability"],
                "lost_units": sim_eval["results"]["expected_lost_units"]
            }

        weighted_service_level = 0.0
        weighted_stockout_prob = 0.0
        total_lost_units = 0.0

        worst_sku = None
        worst_fill = 1.0

        for sku, policy in selected.items():
            weight = (demand_totals[sku] / total_demand_all) if total_demand_all > 0 else 0
            weighted_service_level += weight * sim_risk[sku]["fill_rate"]
            weighted_stockout_prob += weight * sim_risk[sku]["stockout_probability"]
            total_lost_units += sim_risk[sku]["lost_units"]
            if sim_risk[sku]["fill_rate"] < worst_fill:
                worst_fill = sim_risk[sku]["fill_rate"]
                worst_sku = sku

        risk_cols = st.columns(3)
        risk_cols[0].metric("Portfolio Service Level", f"{weighted_service_level:.2%}")
        risk_cols[1].metric("Portfolio Stockout Probability", f"{weighted_stockout_prob:.2%}")
        worst_label = f"{worst_sku} ({worst_fill:.2%})" if worst_sku else "N/A"
        risk_cols[2].metric("Worst-Case SKU Risk", worst_label)

    with st.container():
        st.markdown("### Capital Allocation Across SKUs")

        portfolio_df_sorted = portfolio_df.sort_values(
            "Working Capital Tied Up ($)",
            ascending=False
        ).copy()
        portfolio_df_sorted["Capital % of Total"] = (
            portfolio_df_sorted["Working Capital Tied Up ($)"] / total_capital * 100
            if total_capital > 0 else 0
        )
        portfolio_df_sorted["Bar Label"] = (
            portfolio_df_sorted["Capital % of Total"].map(lambda x: f"{x:.1f}%")
            + " | "
            + portfolio_df_sorted["Working Capital Tied Up ($)"].map(lambda x: f"${x:,.0f}")
        )

        fig_capital = px.bar(
            portfolio_df_sorted,
            x="SKU",
            y="Working Capital Tied Up ($)",
            text=portfolio_df_sorted["Bar Label"],
            title="Working Capital Allocation by SKU ($)"
        )
        fig_capital.update_traces(textposition="outside")
        fig_capital.update_layout(
            template="plotly_dark",
            height=450,
            yaxis_title="Working Capital Tied Up ($)",
            xaxis_title="SKU",
            showlegend=False
        )
        fig_capital.add_hline(y=budget, line_dash="dash", line_color="white")
        st.plotly_chart(fig_capital, use_container_width=True)

    with st.container():
        st.markdown("### Portfolio Demand Forecast Overview")

        all_forecasts = []

        for sku in selected.keys():
            forecast_result = forecast_agent.forecast(
                item_id=sku,
                store_id=store_id,
                start_date=start_date,
                end_date=end_date
            )

            forecast_df = forecast_result["results"][0]["daily_forecast"].copy()
            forecast_df["SKU"] = sku
            all_forecasts.append(forecast_df)

        combined_df = pd.concat(all_forecasts)

        fig = go.Figure()
        # Click legend items to toggle SKUs on/off
        for sku in combined_df["SKU"].unique():
            df_sku = combined_df[combined_df["SKU"] == sku]
            fig.add_trace(go.Scatter(
                x=df_sku["date"],
                y=df_sku["forecast"],
                mode="lines",
                name=sku,
                line=dict(width=2, shape="spline"),
                opacity=0.8
            ))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend=dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0
            ),
            xaxis_title="Date",
            yaxis_title="Units"
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig, use_container_width=True)
