from backend.simulation_agent import SimulationAgent
from backend.optimization_agent import OptimizationAgent


def generate_portfolio_candidates(
    sku_list,
    store_id,
    start_date,
    end_date,
    service_level_target,
    df_base,
    forecast_agent,
    sim_agent,
    opt_agent
):
    s_candidates = range(5, 101, 10)
    Q_candidates = range(10, 81, 10)

    candidate_data = {}

    for sku in sku_list:

        forecast_result = forecast_agent.forecast(
            item_id=sku,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )

        forecast_df = forecast_result["results"][0]["daily_forecast"]

        avg_price = (
            df_base[
                (df_base["item_id"] == sku) &
                (df_base["store_id"] == store_id)
            ]["price"].mean()
        )

        policies = []
        pid = 0

        for s in s_candidates:
            for Q in Q_candidates:

                sim_eval = sim_agent.simulate(
                    forecast_df=forecast_df,
                    item_id=sku,
                    store_id=store_id,
                    s=s,
                    Q=Q,
                    return_sample_path=False,
                    return_full_distribution=False
                )

                fill_rate = sim_eval["results"]["expected_fill_rate"]

                min_fill_rate = max(0.0, service_level_target - 0.02)
                if fill_rate < min_fill_rate:
                    continue

                cost = opt_agent._compute_cost(
                    sim_result=sim_eval,
                    avg_price=avg_price,
                    horizon_days=len(forecast_df)
                )

                investment = (
                    sim_eval["results"]["avg_inventory"] * avg_price
                )

                procurement_spend = (
                    sim_eval["results"]["avg_orders"] * Q * avg_price
                )

                policies.append({
                    "policy_id": pid,
                    "s": s,
                    "Q": Q,
                    "cost": cost["total_cost"],
                    "investment": investment,
                    "procurement_spend": procurement_spend,
                    "fill_rate": fill_rate
                })

                pid += 1

        candidate_data[sku] = policies

    return candidate_data
