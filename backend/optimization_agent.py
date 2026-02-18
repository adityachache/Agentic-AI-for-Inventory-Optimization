import numpy as np

class OptimizationAgent:
    """
    Inventory Optimization Agent
    - Searches over (s, Q)
    - Uses SimulationAgent to evaluate policies
    - Minimizes total cost subject to service constraint
    """

    def __init__(
        self,
        simulation_agent,
        holding_cost_rate=0.25,        # annual holding cost as % of price
        order_cost=20.0,               # fixed cost per order
        stockout_cost_multiplier=3.0,  # lost sales penalty
        target_fill_rate=0.95
    ):
        self.sim_agent = simulation_agent
        self.holding_cost_rate = holding_cost_rate
        self.order_cost = order_cost
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.target_fill_rate = target_fill_rate

    # --------------------------------------------------
    # Cost computation
    # --------------------------------------------------
    def _compute_cost(self, sim_result, avg_price, horizon_days):
        """
        Convert simulation outputs to expected total cost
        """
        # Holding cost (daily)
        daily_holding_cost = (self.holding_cost_rate * avg_price) / 365

        holding_cost = (
            sim_result["results"]["avg_inventory"]
            * daily_holding_cost
            * horizon_days
        )

        ordering_cost = (
            sim_result["results"]["avg_orders"]
            * self.order_cost
        )

        stockout_cost = (
            sim_result["results"]["expected_lost_units"]
            * self.stockout_cost_multiplier
            * avg_price
        )

        total_cost = holding_cost + ordering_cost + stockout_cost

        return {
            "total_cost": float(total_cost),
            "holding_cost": float(holding_cost),
            "ordering_cost": float(ordering_cost),
            "stockout_cost": float(stockout_cost),
        }

    # --------------------------------------------------
    # Main optimization loop
    # --------------------------------------------------
    def optimize(
        self,
        forecast_df,
        item_id,
        store_id,
        avg_price,
        s_candidates,
        Q_candidates
    ):
        horizon_days = len(forecast_df)
        best = None
        all_results = []

        for s in s_candidates:
            for Q in Q_candidates:
                sim_result = self.sim_agent.simulate(
                    forecast_df=forecast_df,
                    item_id=item_id,
                    store_id=store_id,
                    s=s,
                    Q=Q
                )

                fill_rate = sim_result["results"]["expected_fill_rate"]

                # enforce service constraint
                if fill_rate < self.target_fill_rate:
                    continue

                cost = self._compute_cost(
                    sim_result=sim_result,
                    avg_price=avg_price,
                    horizon_days=horizon_days
                )

                record = {
                    "item_id": item_id,
                    "store_id": store_id,
                    "s": s,
                    "Q": Q,
                    "fill_rate": fill_rate,
                    **cost
                }
                all_results.append(record)

                if best is None or cost["total_cost"] < best["total_cost"]:
                    best = record

        return {
            "best_policy": best,
            "all_evaluated": all_results
        }

    def generate_llm_summary(self, llm_client, llm_model, item_payload):
        """
        Generate a managerial summary for a single product.
        item_payload contains ONLY one product's results.
        """

        # Decide wording based on mode
        if item_payload["mode"] == "auto_optimized":
            plan_sentence = (
                "The inventory plan was automatically selected to minimize total cost "
                "while meeting the service level target."
            )
        else:
            plan_sentence = (
                "The inventory plan was manually configured by the user for evaluation."
            )

        # Build the prompt (fully rendered BEFORE sending to LLM)
        prompt = f"""
    You are reviewing inventory planning results for a single product at a retail store.

    Store: {item_payload['store_id']}
    Product: {item_payload['item_id']}
    Planning period: {item_payload['start_date']} to {item_payload['end_date']}
    Service level target: {item_payload['service_level_target']}%

    Demand outlook:
    This product is expected to sell approximately {item_payload['total_units']} units
    during the selected period, averaging about {item_payload['avg_daily_units']:.1f}
    units per day.

    Inventory plan:
    {plan_sentence}

    Recommended stock settings:
    Reorder when inventory falls below {item_payload['s']} units and replenish
    {item_payload['Q']} units at a time. This plan achieves an availability level
    of approximately {item_payload['fill_rate']:.2f}% during the planning period.

    Cost overview for this period:
    Total inventory management cost is estimated at ${item_payload['total_cost']:.2f},
    driven primarily by ordering costs (${item_payload['ordering_cost']:.2f}) and
    inventory holding costs (${item_payload['holding_cost']:.2f}). The expected
    impact of lost sales remains very low at ${item_payload['stockout_cost']:.2f}.

    Instructions:
    Write a concise, non-technical summary for a store manager.
    Explain why this product needs the recommended stock levels and how the plan
    balances availability and cost. End with 2–3 actionable recommendations.

    Constraints:
    - Use plain business language.
    - Do not mention simulations, algorithms, or models.
    - Do not use bullet points.
    - Keep the response under 8 sentences.
    """

        response = llm_client.models.generate_content(
            model=llm_model,
            contents=prompt
        )

        return response.text.strip()