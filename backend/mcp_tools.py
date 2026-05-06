import json
import re
import pandas as pd

from backend.portfolio_optimizer import solve_portfolio


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
def _classify_fill_rate_risk(fill_rate, target_rate):
    """
    Classify policy risk relative to the target service level.
    """
    if fill_rate >= target_rate:
        return "Low Risk"
    if fill_rate >= max(0.0, target_rate - 0.05):
        return "Moderate Risk"
    return "High Risk"


def _compute_portfolio_summary(records, target_fill_rate):
    """
    Aggregate SKU-level results into a portfolio-level shock summary.
    """
    def summarize_policy_field(field_name):
        values = [float(record.get(field_name, 0) or 0) for record in records]
        if not values:
            return "0"
        if all(value == values[0] for value in values):
            return f"{values[0]:.0f}"
        avg_value = sum(values) / len(values)
        return f"Avg {avg_value:.1f} ({min(values):.0f}-{max(values):.0f})"

    total_demand = sum(record.get("total_demand", 0.0) for record in records)
    total_lost_units = sum(record.get("lost_units", 0.0) for record in records)
    fulfilled_units = max(0.0, total_demand - total_lost_units)
    fill_rate = (fulfilled_units / total_demand) if total_demand > 0 else 1.0

    return {
        "sku_count": len(records),
        "s": summarize_policy_field("s"),
        "Q": summarize_policy_field("Q"),
        "fill_rate": float(fill_rate),
        "total_cost": float(sum(record.get("total_cost", 0.0) for record in records)),
        "working_capital": float(sum(record.get("working_capital", 0.0) for record in records)),
        "procurement_spend": float(sum(record.get("procurement_spend", 0.0) for record in records)),
        "lost_units": float(total_lost_units),
        "stockout_days": int(sum(record.get("stockout_days", 0) for record in records)),
        "risk_level": _classify_fill_rate_risk(fill_rate, target_fill_rate),
        "sku_details": records,
        "scope": "portfolio",
    }


def portfolio_event_window_shock_tool(
    portfolio_context,
    portfolio_meta,
    sim_agent,
    opt_agent,
    df_base,
    forecast_agent,
    shock_percentage
):
    """
    Apply a demand shock across all SKUs in the selected portfolio, evaluate the
    current portfolio under shock, then re-optimize the portfolio under the
    shocked demand path and the same budget.
    """
    if not portfolio_context:
        return {"error": "Run portfolio optimization first."}

    store_id = portfolio_meta.get("store_id")
    start_date = portfolio_meta.get("start_date")
    end_date = portfolio_meta.get("end_date")
    budget = float(portfolio_meta.get("budget", 0) or 0)
    target_fill_rate = float(
        portfolio_meta.get("service_level_target", opt_agent.target_fill_rate)
        or opt_agent.target_fill_rate
    )

    s_candidates = range(5, 101, 10)
    Q_candidates = range(10, 81, 10)

    baseline_records = []
    shocked_records = []
    candidate_data = {}
    fallback_warnings = []

    for sku, policy in portfolio_context.items():
        forecast_result = forecast_agent.forecast(
            item_id=sku,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
        baseline_forecast_df = forecast_result["results"][0]["daily_forecast"].copy()
        shocked_forecast_df = baseline_forecast_df.copy()
        shocked_forecast_df["forecast"] = shocked_forecast_df["forecast"] * (1 + shock_percentage)

        avg_price = (
            df_base[
                (df_base["item_id"] == sku) &
                (df_base["store_id"] == store_id)
            ]["price"]
            .mean()
        )
        horizon_days = len(baseline_forecast_df)

        baseline_eval = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=baseline_forecast_df["forecast"].values,
            item_id=sku,
            store_id=store_id,
            s=policy["s"],
            Q=policy["Q"]
        )
        baseline_cost = opt_agent._compute_cost(
            sim_result={
                "results": {
                    "avg_inventory": baseline_eval["avg_inventory"],
                    "avg_orders": baseline_eval["orders"],
                    "expected_lost_units": baseline_eval["lost_units"]
                }
            },
            avg_price=avg_price,
            horizon_days=horizon_days
        )
        baseline_records.append({
            "sku": sku,
            "s": policy["s"],
            "Q": policy["Q"],
            "fill_rate": baseline_eval["fill_rate"],
            "total_cost": baseline_cost["total_cost"],
            "working_capital": float(policy.get("investment", 0) or 0),
            "procurement_spend": float(policy.get("procurement_spend", 0) or 0),
            "lost_units": baseline_eval["lost_units"],
            "stockout_days": baseline_eval["stockout_days"],
            "total_demand": baseline_eval["total_demand"],
        })

        shocked_eval = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=shocked_forecast_df["forecast"].values,
            item_id=sku,
            store_id=store_id,
            s=policy["s"],
            Q=policy["Q"]
        )
        shocked_cost = opt_agent._compute_cost(
            sim_result={
                "results": {
                    "avg_inventory": shocked_eval["avg_inventory"],
                    "avg_orders": shocked_eval["orders"],
                    "expected_lost_units": shocked_eval["lost_units"]
                }
            },
            avg_price=avg_price,
            horizon_days=horizon_days
        )
        shocked_records.append({
            "sku": sku,
            "s": policy["s"],
            "Q": policy["Q"],
            "fill_rate": shocked_eval["fill_rate"],
            "total_cost": shocked_cost["total_cost"],
            "working_capital": float(shocked_eval["avg_inventory"] * avg_price),
            "procurement_spend": float(shocked_eval["orders"] * policy["Q"] * avg_price),
            "lost_units": shocked_eval["lost_units"],
            "stockout_days": shocked_eval["stockout_days"],
            "total_demand": shocked_eval["total_demand"],
        })

        feasible_policies = []
        fallback_policy = None
        policy_id = 0

        for s_val in s_candidates:
            for q_val in Q_candidates:
                sim_eval = sim_agent.evaluate_fixed_demand_scenario(
                    daily_demand=shocked_forecast_df["forecast"].values,
                    item_id=sku,
                    store_id=store_id,
                    s=s_val,
                    Q=q_val
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

                candidate_record = {
                    "policy_id": policy_id,
                    "s": s_val,
                    "Q": q_val,
                    "cost": cost_eval["total_cost"],
                    "investment": float(sim_eval["avg_inventory"] * avg_price),
                    "procurement_spend": float(sim_eval["orders"] * q_val * avg_price),
                    "fill_rate": sim_eval["fill_rate"],
                    "lost_units": sim_eval["lost_units"],
                    "stockout_days": sim_eval["stockout_days"],
                    "total_demand": sim_eval["total_demand"],
                }
                policy_id += 1

                if sim_eval["fill_rate"] >= target_fill_rate:
                    feasible_policies.append(candidate_record)
                else:
                    if (
                        fallback_policy is None
                        or candidate_record["fill_rate"] > fallback_policy["fill_rate"]
                        or (
                            candidate_record["fill_rate"] == fallback_policy["fill_rate"]
                            and candidate_record["cost"] < fallback_policy["cost"]
                        )
                    ):
                        fallback_policy = candidate_record

        if feasible_policies:
            candidate_data[sku] = feasible_policies
        elif fallback_policy is not None:
            candidate_data[sku] = [fallback_policy]
            fallback_warnings.append(sku)
        else:
            return {"error": f"No viable shocked portfolio policy found for SKU {sku}."}

    selected = solve_portfolio(candidate_data, budget)
    if "error" in selected:
        return selected

    optimized_records = []
    for sku, policy in selected.items():
        forecast_result = forecast_agent.forecast(
            item_id=sku,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
        shocked_forecast_df = forecast_result["results"][0]["daily_forecast"].copy()
        shocked_forecast_df["forecast"] = shocked_forecast_df["forecast"] * (1 + shock_percentage)

        avg_price = (
            df_base[
                (df_base["item_id"] == sku) &
                (df_base["store_id"] == store_id)
            ]["price"]
            .mean()
        )
        horizon_days = len(shocked_forecast_df)

        optimized_eval = sim_agent.evaluate_fixed_demand_scenario(
            daily_demand=shocked_forecast_df["forecast"].values,
            item_id=sku,
            store_id=store_id,
            s=policy["s"],
            Q=policy["Q"]
        )
        optimized_cost = opt_agent._compute_cost(
            sim_result={
                "results": {
                    "avg_inventory": optimized_eval["avg_inventory"],
                    "avg_orders": optimized_eval["orders"],
                    "expected_lost_units": optimized_eval["lost_units"]
                }
            },
            avg_price=avg_price,
            horizon_days=horizon_days
        )
        optimized_records.append({
            "sku": sku,
            "s": policy["s"],
            "Q": policy["Q"],
            "fill_rate": optimized_eval["fill_rate"],
            "total_cost": optimized_cost["total_cost"],
            "working_capital": float(policy.get("investment", 0) or 0),
            "procurement_spend": float(policy.get("procurement_spend", 0) or 0),
            "lost_units": optimized_eval["lost_units"],
            "stockout_days": optimized_eval["stockout_days"],
            "total_demand": optimized_eval["total_demand"],
        })

    warning = None
    if fallback_warnings:
        warning = (
            "Some SKUs had no feasible shocked policy at the target service level. "
            f"Fallback policies were used for: {', '.join(sorted(fallback_warnings))}."
        )

    return {
        "analysis_type": "portfolio",
        "baseline_policy": _compute_portfolio_summary(baseline_records, target_fill_rate),
        "shocked_policy": _compute_portfolio_summary(shocked_records, target_fill_rate),
        "shock_optimal_policy": _compute_portfolio_summary(optimized_records, target_fill_rate),
        "shock_percentage": shock_percentage,
        "warning": warning,
    }


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
    Enforces the service level constraint; falls back to the best available
    policy if no feasible policy exists.
    """

    if baseline_context is None:
        return {"error": "Run dashboard optimization first."}

    # Extract baseline inputs
    store_id = baseline_context["inputs"]["store_id"]
    item_id = baseline_context["inputs"]["item_id"]
    start_date = baseline_context["inputs"]["start_date"]
    end_date = baseline_context["inputs"]["end_date"]

    baseline_policy = baseline_context["policies"][0]
    baseline_s = baseline_policy["s"]
    target_fill_rate = float(
        baseline_context.get("assumptions", {}).get(
            "service_level_target",
            opt_agent.target_fill_rate
        )
        or opt_agent.target_fill_rate
    )

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

    horizon_days = len(shocked_df)

    # --------------------------------------------------
    # 3a. Evaluate baseline policy under shocked demand
    # --------------------------------------------------
    shocked_eval = sim_agent.evaluate_fixed_demand_scenario(
        daily_demand=shocked_df["forecast"].values,
        item_id=item_id,
        store_id=store_id,
        s=baseline_policy["s"],
        Q=baseline_policy["Q"]
    )

    shocked_cost = opt_agent._compute_cost(
        sim_result={
            "results": {
                "avg_inventory": shocked_eval["avg_inventory"],
                "avg_orders": shocked_eval["orders"],
                "expected_lost_units": shocked_eval["lost_units"]
            }
        },
        avg_price=avg_price,
        horizon_days=horizon_days
    )

    def _classify_shocked_risk(fill_rate):
        if fill_rate >= 0.95:
            return "Low Risk"
        if fill_rate >= 0.90:
            return "Moderate Risk"
        return "High Risk"

    shocked_policy = {
        "s": baseline_policy["s"],
        "Q": baseline_policy["Q"],
        "fill_rate": shocked_eval["fill_rate"],
        "total_cost": shocked_cost["total_cost"],
        "lost_units": shocked_eval["lost_units"],
        "stockout_days": shocked_eval["stockout_days"],
        "risk_level": _classify_shocked_risk(shocked_eval["fill_rate"])
    }

    best_feasible = None
    best_fallback = None
    best_fallback_prefer_baseline = None

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

            # Enforce service level constraint for feasible policies
            if record["fill_rate"] >= target_fill_rate:
                if best_feasible is None or record["total_cost"] < best_feasible["total_cost"]:
                    best_feasible = record
            else:
                # Track best available policy if none meet target
                if (
                    best_fallback is None
                    or record["fill_rate"] > best_fallback["fill_rate"]
                    or (
                        record["fill_rate"] == best_fallback["fill_rate"]
                        and record["total_cost"] < best_fallback["total_cost"]
                    )
                ):
                    best_fallback = record
                # Prefer non-decreasing reorder points under shock when infeasible
                if s_val >= baseline_s:
                    if (
                        best_fallback_prefer_baseline is None
                        or record["fill_rate"] > best_fallback_prefer_baseline["fill_rate"]
                        or (
                            record["fill_rate"] == best_fallback_prefer_baseline["fill_rate"]
                            and record["total_cost"] < best_fallback_prefer_baseline["total_cost"]
                        )
                    ):
                        best_fallback_prefer_baseline = record

    warning = None
    if best_feasible is None:
        # If nothing meets target, return the best available policy
        best_shock = best_fallback_prefer_baseline or best_fallback
        warning = (
            "No policy met the target service level under shock. "
            "Returning the best available policy; risk is elevated."
        )
    else:
        best_shock = best_feasible

    if best_shock is None:
        return {"error": "No viable shock policy found."}

    baseline_risk = _classify_fill_rate_risk(baseline_policy["fill_rate"], target_fill_rate)
    best_shock["risk_level"] = _classify_fill_rate_risk(best_shock["fill_rate"], target_fill_rate)

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
        "shocked_policy": shocked_policy,
        "shock_optimal_policy": best_shock,
        "shock_percentage": shock_percentage,
        "warning": warning,
        "baseline_forecast": baseline_forecast_df,
        "shocked_forecast": shocked_df
    }
