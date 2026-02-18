from backend.simulation_agent import SimulationAgent
from backend.optimization_agent import OptimizationAgent
import pandas as pd

DEFAULT_ASSUMPTIONS = {
    "lead_time_days": 2,
    "order_cost": 20.0,
    "holding_cost_rate": 0.25,
    "stockout_cost_multiplier": 3.0,
    "demand_cv": 0.25,
    "n_simulations": 500
}

def run_inventory_planner(
    store_id,
    item_ids,
    start_date,
    end_date,
    service_level_target=0.95,
    mode="auto",                 # "auto" or "manual"
    manual_policy=None,          # {"s": ..., "Q": ...}
    assumptions=None,
    df_base=None,
    forecast_agent=None,
    llm_client=None,
    llm_model="gemini-3-flash-preview"
):
    """
    Main end-to-end inventory planning pipeline.
    """

    # --------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------
    if df_base is None:
        raise ValueError("df_base must be provided (build it once and reuse it).")

    if forecast_agent is None:
        raise ValueError("forecast_agent must be provided.")

    if not item_ids:
        raise ValueError("At least one product must be selected.")

    if mode == "manual" and manual_policy is None:
        raise ValueError("manual_policy must be provided when mode='manual'.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # --------------------------------------------------
    # 2. Merge assumptions with defaults
    # --------------------------------------------------
    cfg = DEFAULT_ASSUMPTIONS.copy()
    if assumptions:
        cfg.update(assumptions)

    # --------------------------------------------------
    # 3. Initialize agents (simulation + optimization)
    # --------------------------------------------------
    sim_agent = SimulationAgent(
        n_simulations=cfg["n_simulations"],
        demand_cv=cfg["demand_cv"],
        lead_time_days=cfg["lead_time_days"]
    )

    opt_agent = OptimizationAgent(
        simulation_agent=sim_agent,
        holding_cost_rate=cfg["holding_cost_rate"],
        order_cost=cfg["order_cost"],
        stockout_cost_multiplier=cfg["stockout_cost_multiplier"],
        target_fill_rate=service_level_target
    )

    # --------------------------------------------------
    # 4. Forecast demand (NO LLM HERE)
    # --------------------------------------------------
    forecast_out = forecast_agent.forecast(
        item_ids=item_ids,
        store_id=store_id,
        start_date=start_date,
        end_date=end_date
    )

    forecast_results = {}
    for r in forecast_out["results"]:
        forecast_results[r["item_id"]] = {
            "daily_forecast": r["daily_forecast"],
            "total_units": r["total_units"],
            "avg_daily_units": r["daily_forecast"]["forecast"].mean()
        }

    # --------------------------------------------------
    # 5. Optimize or simulate policies
    # --------------------------------------------------
    policy_results = []

    for item_id in item_ids:
        forecast_df = forecast_results[item_id]["daily_forecast"]

        # average price for cost calculation
        avg_price = (
            df_base[
                (df_base["item_id"] == item_id) &
                (df_base["store_id"] == store_id)
            ]["price"]
            .mean()
        )

        if mode == "auto":
            # candidate search space (simple & explainable)
            s_candidates = range(50, 251, 25)
            Q_candidates = range(100, 501, 50)

            opt_result = opt_agent.optimize(
                forecast_df=forecast_df,
                item_id=item_id,
                store_id=store_id,
                avg_price=avg_price,
                s_candidates=s_candidates,
                Q_candidates=Q_candidates
            )

            best = opt_result["best_policy"]

        else:  # manual mode
            s = manual_policy["s"]
            Q = manual_policy["Q"]

            sim_result = sim_agent.simulate(
                forecast_df=forecast_df,
                item_id=item_id,
                store_id=store_id,
                s=s,
                Q=Q
            )

            cost = opt_agent._compute_cost(
                sim_result=sim_result,
                avg_price=avg_price,
                horizon_days=len(forecast_df)
            )

            best = {
                "item_id": item_id,
                "store_id": store_id,
                "s": s,
                "Q": Q,
                "fill_rate": sim_result["results"]["expected_fill_rate"],
                **cost
            }

        sim_result = sim_agent.simulate(
            forecast_df=forecast_df,
            item_id=item_id,
            store_id=store_id,
            s=best["s"],
            Q=best["Q"]
        )

        scenario_summary = sim_result["scenario_summary"]

        # Deterministic stress testing
        low_scenario = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=scenario_summary["p10_daily"],
            item_id=item_id,
            store_id=store_id,
            s=best["s"],
            Q=best["Q"]
        )

        base_scenario = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=scenario_summary["p50_daily"],
            item_id=item_id,
            store_id=store_id,
            s=best["s"],
            Q=best["Q"]
        )

        high_scenario = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=scenario_summary["p90_daily"],
            item_id=item_id,
            store_id=store_id,
            s=best["s"],
            Q=best["Q"]
        )

        # Attach scenarios to best policy
        best["scenario_analysis"] = {
            "confidence_interval": {
                "p10_total_demand": scenario_summary["p10_total_demand"],
                "p50_total_demand": scenario_summary["p50_total_demand"],
                "p90_total_demand": scenario_summary["p90_total_demand"],
            },
            "low": low_scenario,
            "base": base_scenario,
            "high": high_scenario
        }

        policy_results.append(best)

    # --------------------------------------------------
    # 6. Build LLM payload (AGGREGATED ONLY)
    # --------------------------------------------------
    products_payload = []
    for p in policy_results:
        f = forecast_results[p["item_id"]]
        products_payload.append({
            "item_id": p["item_id"],
            "total_units": int(f["total_units"]),
            "avg_daily_units": round(f["avg_daily_units"], 1),
            "s": p["s"],
            "Q": p["Q"],
            "fill_rate": round(p["fill_rate"] * 100, 2),
            "total_cost": round(p["total_cost"], 2),
            "holding_cost": round(p["holding_cost"], 2),
            "ordering_cost": round(p["ordering_cost"], 2),
            "stockout_cost": round(p["stockout_cost"], 2),
        })

    # llm_payload = {
    #     "store_id": store_id,
    #     "start_date": str(start_date.date()),
    #     "end_date": str(end_date.date()),
    #     "service_level_target": int(service_level_target * 100),
    #     "mode": "auto_optimized" if mode == "auto" else "manual_policy",
    #     "products": products_payload
    # }

    # --------------------------------------------------
    # 7. Final LLM summary (ONE CALL)
    # --------------------------------------------------
    # --------------------------------------------------
# 7. Final LLM summaries (ONE PER ITEM)
# --------------------------------------------------
    llm_summaries = {}

    if llm_client is not None:
        for p in products_payload:
            # Build per-item payload
            item_payload = {
                "store_id": store_id,
                "item_id": p["item_id"],
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "service_level_target": int(service_level_target * 100),
                "mode": "auto_optimized" if mode == "auto" else "manual_policy",
                "total_units": p["total_units"],
                "avg_daily_units": p["avg_daily_units"],
                "s": p["s"],
                "Q": p["Q"],
                "fill_rate": p["fill_rate"],
                "total_cost": p["total_cost"],
                "holding_cost": p["holding_cost"],
                "ordering_cost": p["ordering_cost"],
                "stockout_cost": p["stockout_cost"]
            }

            summary = opt_agent.generate_llm_summary(
                llm_client=llm_client,
                llm_model=llm_model,
                item_payload=item_payload
            )

            llm_summaries[p["item_id"]] = summary
    else:
        llm_summaries = {
            p["item_id"]: "LLM summary disabled."
            for p in products_payload
        }

    # --------------------------------------------------
    # 8. Final return object (for Streamlit)
    # --------------------------------------------------
    return {
        "inputs": {
            "store_id": store_id,
            "item_ids": item_ids,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "service_level_target": service_level_target,
            "mode": mode
        },
        "forecast": forecast_results,
        "policies": policy_results,
        "llm_summaries": llm_summaries,
        "assumptions": cfg,
        "scenario_analysis": policy_results[best['scenario_analysis']]
    }