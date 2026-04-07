# backend/dashboard_controller.py
import numpy as np

def run_single_sku_dashboard(
    store_id,
    item_id,
    start_date,
    end_date,
    service_level_target,
    mode,
    manual_policy,
    assumptions,
    df_base,
    forecast_agent,
    llm_client
):
    from backend.agent_pipeline import run_inventory_planner

    results = run_inventory_planner(
        store_id=store_id,
        item_id=item_id,
        start_date=start_date,
        end_date=end_date,
        service_level_target=service_level_target,
        mode=mode,
        manual_policy=manual_policy,
        assumptions=assumptions,
        df_base=df_base,
        forecast_agent=forecast_agent,
        llm_client=llm_client
    )

    if results and results.get("policies") and results.get("simulation_path"):
        results["policies"][0]["simulation_path"] = results["simulation_path"]

    return results


def run_simulation_preview(
    store_id,
    item_id,
    start_date,
    end_date,
    df_base,
    forecast_agent,
    policy=None,
    assumptions=None
):
    from backend.simulation_agent import SimulationAgent
    from backend.agent_pipeline import DEFAULT_ASSUMPTIONS

    forecast_result = forecast_agent.forecast(
        item_id=item_id,
        store_id=store_id,
        start_date=start_date,
        end_date=end_date
    )

    forecast_df = forecast_result["results"][0]["daily_forecast"]

    if policy is not None:
        s_mid = policy["s"]
        Q_mid = policy["Q"]
    else:
        s_candidates = list(range(5, 101, 10))
        Q_candidates = list(range(10, 81, 10))
        s_mid = s_candidates[len(s_candidates) // 2]
        Q_mid = Q_candidates[len(Q_candidates) // 2]

    cfg = DEFAULT_ASSUMPTIONS.copy()
    if assumptions:
        cfg.update({
            "lead_time_days": assumptions.get("lead_time_days", cfg["lead_time_days"]),
            "demand_cv": assumptions.get("demand_cv", cfg["demand_cv"]),
            "n_simulations": assumptions.get("n_simulations", cfg["n_simulations"])
        })

    sim_agent = SimulationAgent(
        n_simulations=cfg["n_simulations"],
        demand_cv=cfg["demand_cv"],
        lead_time_days=cfg["lead_time_days"]
    )
    sim_result = sim_agent.simulate(
        forecast_df=forecast_df,
        item_id=item_id,
        store_id=store_id,
        s=s_mid,
        Q=Q_mid,
        return_sample_path=True,
        return_full_distribution=True
    )

    preview = sim_result["monte_carlo_preview"]
    preview["sample_path"] = sim_result["sample_path"]
    fill_rates = np.array(preview.get("fill_rate", []), dtype=float)
    lost_units = np.array(preview.get("lost_units", []), dtype=float)
    stockout_days = np.array(preview.get("stockout_days", []), dtype=float)

    preview.update({
        "mean_fill_rate": float(np.mean(fill_rates)) if fill_rates.size else 0.0,
        "p5_fill_rate": float(np.percentile(fill_rates, 5)) if fill_rates.size else 0.0,
        "p95_fill_rate": float(np.percentile(fill_rates, 95)) if fill_rates.size else 0.0,
        "stockout_probability": float(np.mean(stockout_days > 0)) if stockout_days.size else 0.0,
        "mean_lost_units": float(np.mean(lost_units)) if lost_units.size else 0.0,
        "worst_case_lost_units": float(np.max(lost_units)) if lost_units.size else 0.0
    })

    return preview
