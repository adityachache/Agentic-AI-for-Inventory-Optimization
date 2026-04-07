import json
import re
import pandas as pd


# ---------------------------
# Forecast Tool
# ---------------------------
def forecast_tool(context, forecast_agent, store_id, item_id, start_date, end_date):
    result = forecast_agent.forecast(
        item_id=item_id,
        store_id=store_id,
        start_date=start_date,
        end_date=end_date
    )
    context.update("forecast", result)
    return result


# ---------------------------
# Optimize Tool
# ---------------------------
def optimize_tool(context, run_inventory_planner, inputs):
    result = run_inventory_planner(**inputs)
    context.update("optimization", result)
    return result


# ---------------------------
# Intent Parser (LLM)
# ---------------------------
def parse_intent_with_llm(llm_client, llm_model, user_prompt):

    prompt = f"""
Extract structured JSON from this request.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.

Keys:
- action (forecast / optimize / shock)
- shock_percentage (required only if action=shock)

User request:
{user_prompt}
"""

    response = llm_client.models.generate_content(
        model=llm_model,
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown fences if present
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    try:
        return json.loads(text)
    except:
        print("Raw LLM response:", text)
        return None


# ---------------------------
# Full-Horizon Shock Tool
# ---------------------------
def event_window_shock_tool(
    baseline_context,
    sim_agent,
    opt_agent,
    df_base,
    forecast_agent,
    shock_percentage
):
    """
    Apply demand shock across full forecast horizon and re-optimize policy.
    """

    if baseline_context is None:
        return {"error": "Run dashboard optimization first."}

    # Extract baseline inputs
    store_id = baseline_context["inputs"]["store_id"]
    item_id = baseline_context["inputs"]["item_id"]
    start_date = baseline_context["inputs"]["start_date"]
    end_date = baseline_context["inputs"]["end_date"]

    baseline_policy = baseline_context["policies"][0]

    # --------------------------------------------------
    # 1. Get baseline forecast
    # --------------------------------------------------
    forecast_result = forecast_agent.forecast(
        item_id=item_id,
        store_id=store_id,
        start_date=start_date,
        end_date=end_date
    )

    baseline_forecast_df = forecast_result["results"][0]["daily_forecast"].copy()

    # --------------------------------------------------
    # 2. Apply shock to full horizon
    # --------------------------------------------------
    shocked_df = baseline_forecast_df.copy()
    shocked_df["forecast"] = shocked_df["forecast"] * (1 + shock_percentage)

    # --------------------------------------------------
    # 3. Re-optimize under shocked demand
    # --------------------------------------------------
    avg_price = (
        df_base[
            (df_base["item_id"] == item_id) &
            (df_base["store_id"] == store_id)
        ]["price"]
        .mean()
    )

    s_candidates = range(5, 101, 10)
    Q_candidates = range(10, 81, 10)

    best_shock = None
    horizon_days = len(shocked_df)

    for s_val in s_candidates:
        for Q_val in Q_candidates:

            sim_eval = sim_agent.evaluate_fixed_demand_scenario(
                daily_demand=shocked_df["forecast"].values,
                item_id=item_id,
                store_id=store_id,
                s=s_val,
                Q=Q_val
            )

            cost_eval = opt_agent._compute_cost(
                sim_result={
                    "results": {
                        "avg_inventory": sim_eval["avg_inventory"],
                        "avg_orders": sim_eval["orders"],
                        "expected_lost_units": sim_eval["lost_units"]
                    }
                },
                avg_price=avg_price,
                horizon_days=horizon_days
            )

            record = {
                "s": s_val,
                "Q": Q_val,
                "fill_rate": sim_eval["fill_rate"],
                "total_cost": cost_eval["total_cost"],
                "lost_units": sim_eval["lost_units"],
                "stockout_days": sim_eval["stockout_days"]
            }

            if best_shock is None or record["total_cost"] < best_shock["total_cost"]:
                best_shock = record

    def _classify_fill_rate_risk(fill_rate):
        if fill_rate >= 0.98:
            return "Low Risk"
        if fill_rate >= 0.95:
            return "Moderate Risk"
        return "High Risk"

    baseline_risk = _classify_fill_rate_risk(baseline_policy["fill_rate"])
    best_shock["risk_level"] = _classify_fill_rate_risk(best_shock["fill_rate"])

    # --------------------------------------------------
    # 4. Return structured comparison
    # --------------------------------------------------
    return {
        "baseline_policy": {
            "s": baseline_policy["s"],
            "Q": baseline_policy["Q"],
            "fill_rate": baseline_policy["fill_rate"],
            "total_cost": baseline_policy["total_cost"],
            "risk_level": baseline_risk
        },
        "shock_optimal_policy": best_shock,
        "shock_percentage": shock_percentage,
        "baseline_forecast": baseline_forecast_df,
        "shocked_forecast": shocked_df
    }
