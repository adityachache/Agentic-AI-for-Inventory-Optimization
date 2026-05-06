# backend/ui_renderers.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_forecast(results):
    st.subheader("Demand Forecast & Confidence")

    view_mode = st.radio(
        "Forecast View",
        ["Daily Demand", "Cumulative Demand"],
        horizontal=True
    )

    for policy in results["policies"]:
        item_id = policy["item_id"]
        scenario_data = policy.get("scenario_analysis", {})
        ci = scenario_data.get("confidence_interval", {})

        forecast_df = results["forecast"][item_id]["daily_forecast"].copy()
        forecast_df = forecast_df.set_index("date")
        forecast_df["cumulative"] = forecast_df["forecast"].cumsum()

        dates = forecast_df.index
        mean_forecast = forecast_df["forecast"].values

        mean_demand = forecast_df["forecast"].mean()
        std_demand = forecast_df["forecast"].std()

        if mean_demand > 0:
            cv = std_demand / mean_demand
        else:
            cv = 0

        # Classification thresholds
        if mean_demand < 0.5:
            volatility_label = "Intermittent Demand"
            color = "gray"
        elif cv < 0.3:
            volatility_label = "Low Volatility"
            color = "green"
        elif cv < 0.6:
            volatility_label = "Medium Volatility"
            color = "orange"
        else:
            volatility_label = "High Volatility"
            color = "red"

        if "p10_daily" in scenario_data and "p90_daily" in scenario_data:
            p10 = np.array(scenario_data["p10_daily"])
            p90 = np.array(scenario_data["p90_daily"])
        else:
            p10 = mean_forecast
            p90 = mean_forecast

        st.markdown(f"### {item_id}")
        st.markdown(
            f"**Demand Volatility:** "
            f"<span style='color:{color}'>{volatility_label}</span> "
            f"(CV = {cv:.2f})",
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(p90) + list(p10[::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="90% Confidence Band"
        ))

        if view_mode == "Daily Demand":

            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df["forecast"],
                mode="lines",
                name="Daily Forecast",
                line=dict(width=3)
            ))

        else:

            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df["cumulative"],
                mode="lines",
                name="Cumulative Demand",
                line=dict(width=3)
            ))

        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Units"
        )

        st.plotly_chart(fig, use_container_width=True)

        if ci:
            st.caption(
                f"Expected demand: {ci['p50_total_demand']:.0f} units "
                f"(90% range: {ci['p10_total_demand']:.0f} – {ci['p90_total_demand']:.0f})"
            )


def render_simulation_storyboard(results):
    st.subheader("Simulation Storyboard")

    if not results or not results.get("policies"):
        st.info("No simulation data available.")
        return

    policy = results["policies"][0]
    sim_path = policy.get("simulation_path")
    if not sim_path:
        st.info("Simulation path not available.")
        return

    item_id = policy["item_id"]
    forecast_df = results["forecast"][item_id]["daily_forecast"].copy()
    dates = pd.to_datetime(forecast_df["date"]).tolist()

    inventory_path = sim_path.get("inventory_path", [])
    demand_path = sim_path.get("demand_path", [])
    stockout_flags = sim_path.get("stockout_days", [])
    reorder_days = sim_path.get("reorder_days", [])
    arrival_days = sim_path.get("arrival_days", [])

    min_len = min(
        len(dates),
        len(inventory_path),
        len(demand_path),
        len(stockout_flags)
    )
    dates = dates[:min_len]
    inventory_path = inventory_path[:min_len]
    demand_path = demand_path[:min_len]
    stockout_flags = stockout_flags[:min_len]

    sim_metrics = policy.get("simulation_metrics", {})
    fill_rate = sim_metrics.get("fill_rate", policy.get("fill_rate"))
    lost_units = sim_metrics.get("lost_units")
    stockout_days = sim_metrics.get("stockout_days")
    avg_inventory = sim_metrics.get("avg_inventory")
    orders_placed = sim_metrics.get("orders_placed")

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Fill Rate", f"{(fill_rate or 0):.2%}")
    kpi_cols[1].metric("Lost Units", f"{(lost_units or 0):.1f}")
    kpi_cols[2].metric("Stockout Days", f"{(stockout_days or 0):.1f}")
    kpi_cols[3].metric("Average Inventory", f"{(avg_inventory or 0):.1f}")
    kpi_cols[4].metric("Orders Placed", f"{(orders_placed or 0):.1f}")

    if fill_rate is None:
        risk_label = "Unknown Risk"
        color = "gray"
    elif fill_rate >= 0.98:
        risk_label = "Low Risk"
        color = "green"
    elif fill_rate >= 0.95:
        risk_label = "Moderate Risk"
        color = "orange"
    else:
        risk_label = "High Risk"
        color = "red"

    st.markdown(
        f"**Risk Level:** <span style='color:{color}; font-weight:600'>{risk_label}</span>",
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=inventory_path,
        mode="lines",
        name="Inventory Level",
        line=dict(width=3)
    ))

    stockout_dates = [dates[i] for i, flag in enumerate(stockout_flags) if flag]
    stockout_inv = [inventory_path[i] for i, flag in enumerate(stockout_flags) if flag]
    if stockout_dates:
        fig.add_trace(go.Scatter(
            x=stockout_dates,
            y=stockout_inv,
            mode="markers",
            name="Stockout Day",
            marker=dict(color="red", size=8)
        ))

    reorder_dates = [dates[i] for i in reorder_days if i < len(dates)]
    reorder_inv = [inventory_path[i] for i in reorder_days if i < len(inventory_path)]
    if reorder_dates:
        fig.add_trace(go.Scatter(
            x=reorder_dates,
            y=reorder_inv,
            mode="markers",
            name="Reorder Placed",
            marker=dict(color="orange", size=8, symbol="triangle-up")
        ))

    arrival_dates = [dates[i] for i in arrival_days if i < len(dates)]
    arrival_inv = [inventory_path[i] for i in arrival_days if i < len(inventory_path)]
    if arrival_dates:
        fig.add_trace(go.Scatter(
            x=arrival_dates,
            y=arrival_inv,
            mode="markers",
            name="Order Arrived",
            marker=dict(color="green", size=8, symbol="circle")
        ))

    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Inventory Level"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_policy_tables(results):

    st.subheader("Recommended Inventory Policy")

    policy_df = pd.DataFrame(results["policies"])
    st.dataframe(
        policy_df[["item_id", "s", "Q", "fill_rate", "total_cost"]],
        use_container_width=True
    )

    st.markdown("**Total cost = ordering cost + holding cost + stockout cost.**")
    st.subheader("Cost Breakdown")

    st.dataframe(
        policy_df[["item_id", "holding_cost", "ordering_cost", "stockout_cost"]],
        use_container_width=True
    )


def render_scenarios(results):

    st.subheader("Scenario Stress Test (Deterministic Percentile Paths)")
    st.caption(
        "Deterministic replay of P10/P50/P90 demand paths derived from the Monte Carlo simulation."
    )

    for policy in results["policies"]:
        scenario = policy.get("scenario_analysis", {})
        if not scenario:
            continue

        st.markdown(f"### {policy['item_id']}")

        scenario_table = pd.DataFrame([
            {
                "Scenario": "Low (P10)",
                "Total Demand": scenario["low"]["total_demand"],
                "Fill Rate": scenario["low"]["fill_rate"],
                "Lost Units": scenario["low"]["lost_units"],
                "Stockout Days": scenario["low"]["stockout_days"]
            },
            {
                "Scenario": "Base (P50)",
                "Total Demand": scenario["base"]["total_demand"],
                "Fill Rate": scenario["base"]["fill_rate"],
                "Lost Units": scenario["base"]["lost_units"],
                "Stockout Days": scenario["base"]["stockout_days"]
            },
            {
                "Scenario": "High (P90)",
                "Total Demand": scenario["high"]["total_demand"],
                "Fill Rate": scenario["high"]["fill_rate"],
                "Lost Units": scenario["high"]["lost_units"],
                "Stockout Days": scenario["high"]["stockout_days"]
            }
        ])

        st.dataframe(scenario_table, use_container_width=True)


def render_optimal_scenarios(results):

    st.subheader("Optimal Policies Under Demand Scenarios")

    for policy in results["policies"]:
        scenario_opts = policy.get("scenario_optimal_policies", {})
        if not scenario_opts:
            continue

        st.markdown(f"### {policy['item_id']}")

        opt_table = pd.DataFrame([
            {
                "Scenario": "Low",
                "s": scenario_opts["low"]["s"],
                "Q": scenario_opts["low"]["Q"],
                "Cost": scenario_opts["low"]["total_cost"]
            },
            {
                "Scenario": "Base",
                "s": scenario_opts["base"]["s"],
                "Q": scenario_opts["base"]["Q"],
                "Cost": scenario_opts["base"]["total_cost"]
            },
            {
                "Scenario": "High",
                "s": scenario_opts["high"]["s"],
                "Q": scenario_opts["high"]["Q"],
                "Cost": scenario_opts["high"]["total_cost"]
            }
        ])

        st.dataframe(opt_table, use_container_width=True)
