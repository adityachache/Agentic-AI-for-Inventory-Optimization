import numpy as np
import pandas as pd

class SimulationAgent:
    """
    Monte Carlo Inventory Simulation Agent
    - Simulates demand uncertainty
    - Evaluates an (s, Q) inventory policy
    """

    def __init__(
        self,
        n_simulations: int = 500,
        demand_cv: float = 0.25,
        lead_time_days: int = 2,
        random_seed: int = 42
    ):
        self.n_simulations = n_simulations
        self.demand_cv = demand_cv
        self.lead_time_days = lead_time_days
        np.random.seed(random_seed)

    def simulate(
        self,
        forecast_df: pd.DataFrame,
        item_id: str,
        store_id: str,
        s: int,
        Q: int
    ):
        daily_means = forecast_df["forecast"].values
        n_days = len(daily_means)

        daily_sigma = np.maximum(1.0, self.demand_cv * daily_means)
        initial_inventory = int(s)

        fill_rates = []
        stockout_flags = []
        stockout_days = []
        lost_units = []
        avg_inventory = []
        total_demand_list = []
        orders_list = []

        # NEW: store simulated daily demand paths
        daily_demand_matrix = []

        for sim in range(self.n_simulations):
            inventory = initial_inventory
            on_order = []
            orders_count = 0

            total_demand = 0
            fulfilled = 0
            lost = 0
            inv_levels = []
            stockout_day_count = 0

            daily_demands = []

            for day in range(n_days):
                arrivals = [q for (d, q) in on_order if d == day]
                inventory += sum(arrivals)
                on_order = [(d, q) for (d, q) in on_order if d != day]

                demand = np.random.normal(daily_means[day], daily_sigma[day])
                demand = max(0, demand)

                daily_demands.append(demand)
                total_demand += demand

                if inventory >= demand:
                    inventory -= demand
                    fulfilled += demand
                else:
                    fulfilled += inventory
                    lost += (demand - inventory)
                    inventory = 0
                    stockout_day_count += 1

                inventory_position = inventory + sum(q for (_, q) in on_order)
                if inventory_position <= s:
                    on_order.append((day + self.lead_time_days, Q))
                    orders_count += 1

                inv_levels.append(inventory)

            fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0

            fill_rates.append(fill_rate)
            stockout_flags.append(stockout_day_count > 0)
            stockout_days.append(stockout_day_count)
            lost_units.append(lost)
            avg_inventory.append(np.mean(inv_levels))
            total_demand_list.append(total_demand)
            orders_list.append(orders_count)

            daily_demand_matrix.append(daily_demands)

        # Convert to numpy for percentile calculations
        daily_demand_matrix = np.array(daily_demand_matrix)

        return {
            "item_id": item_id,
            "store_id": store_id,
            "policy": {"s": s, "Q": Q},
            "lead_time_days": self.lead_time_days,
            "initial_inventory": initial_inventory,
            "results": {
                "expected_fill_rate": float(np.mean(fill_rates)),
                "stockout_probability": float(np.mean(stockout_flags)),
                "avg_stockout_days": float(np.mean(stockout_days)),
                "expected_lost_units": float(np.mean(lost_units)),
                "avg_inventory": float(np.mean(avg_inventory)),
                "avg_orders": float(np.mean(orders_list)),
            },
            "scenario_summary": {
                "mean_total_demand": float(np.mean(total_demand_list)),
                "p10_total_demand": float(np.percentile(total_demand_list, 10)),
                "p50_total_demand": float(np.percentile(total_demand_list, 50)),
                "p90_total_demand": float(np.percentile(total_demand_list, 90)),

                # NEW: daily demand confidence bands
                "p10_daily": np.percentile(daily_demand_matrix, 10, axis=0).tolist(),
                "p50_daily": np.percentile(daily_demand_matrix, 50, axis=0).tolist(),
                "p90_daily": np.percentile(daily_demand_matrix, 90, axis=0).tolist(),
            }
        }
    
    def evaluate_fixed_demand_scenario(
        self,
        daily_demand,
        item_id,
        store_id,
        s,
        Q
    ):
        """
        Evaluate inventory performance under a fixed daily demand path
        (no randomness).
        """

        n_days = len(daily_demand)
        inventory = int(s)
        on_order = []
        orders_count = 0

        total_demand = 0
        fulfilled = 0
        lost = 0
        inv_levels = []
        stockout_day_count = 0

        for day in range(n_days):
            arrivals = [q for (d, q) in on_order if d == day]
            inventory += sum(arrivals)
            on_order = [(d, q) for (d, q) in on_order if d != day]

            demand = daily_demand[day]
            total_demand += demand

            if inventory >= demand:
                inventory -= demand
                fulfilled += demand
            else:
                fulfilled += inventory
                lost += (demand - inventory)
                inventory = 0
                stockout_day_count += 1

            inventory_position = inventory + sum(q for (_, q) in on_order)
            if inventory_position <= s:
                on_order.append((day + self.lead_time_days, Q))
                orders_count += 1

            inv_levels.append(inventory)

        fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0

        return {
            "item_id": item_id,
            "store_id": store_id,
            "total_demand": float(total_demand),
            "fill_rate": float(fill_rate),
            "lost_units": float(lost),
            "stockout_days": int(stockout_day_count),
            "avg_inventory": float(np.mean(inv_levels)),
            "orders": int(orders_count)
        }