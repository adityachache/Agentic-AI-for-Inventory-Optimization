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
        demand_cv: float = 0.8,
        lead_time_days: int = 10,
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
        Q: int,
        return_sample_path: bool = True,
        return_full_distribution: bool = True
    ):
        daily_means = np.maximum(0.0, forecast_df["forecast"].values.astype(float))
        n_days = len(daily_means)
        initial_inventory = s
        mean_daily_demand = float(np.mean(daily_means)) if n_days > 0 else 0.0

        fill_rates = []
        stockout_flags = []
        stockout_days = []
        lost_units = []
        avg_inventory = []
        total_demand_list = []
        orders_list = []

        # NEW: store simulated daily demand paths
        daily_demand_matrix = []
        sample_inventory_path = []
        sample_demand_path = []
        sample_stockout_days = []
        sample_reorder_days = []
        sample_arrival_days = []
        preview_inventory_paths = []

        for sim in range(self.n_simulations):
            inventory = initial_inventory
            on_order = []
            orders_count = 0

            total_demand = 0
            fulfilled = 0
            lost = 0
            inv_levels = []
            stockout_day_count = 0

            if mean_daily_demand < 1:
                daily_demands = np.random.poisson(np.maximum(daily_means, 0.001))
            else:
                daily_demands = np.maximum(
                    0,
                    np.round(
                        np.random.normal(
                            daily_means,
                            daily_means * self.demand_cv
                        )
                    )
                ).astype(int)

            for day in range(n_days):
                arrivals = [q for (d, q) in on_order if d == day]
                inventory += sum(arrivals)
                on_order = [(d, q) for (d, q) in on_order if d != day]

                demand = int(daily_demands[day])
                total_demand += demand

                stockout_today = inventory < demand

                if inventory >= demand:
                    inventory -= demand
                    fulfilled += demand
                else:
                    fulfilled += inventory
                    lost += (demand - inventory)
                    inventory = 0
                    stockout_day_count += 1

                inventory_position = inventory + sum(q for (_, q) in on_order)
                if day > 0 and inventory_position <= s:
                    on_order.append((day + self.lead_time_days, Q))
                    orders_count += 1

                inv_levels.append(inventory)

                if return_sample_path and sim == 0:
                    if arrivals:
                        sample_arrival_days.append(day)
                    if day > 0 and inventory_position <= s:
                        sample_reorder_days.append(day)
                    sample_inventory_path.append(inventory)
                    sample_demand_path.append(demand)
                    sample_stockout_days.append(stockout_today)

            fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0

            fill_rates.append(fill_rate)
            stockout_flags.append(stockout_day_count > 0)
            stockout_days.append(stockout_day_count)
            lost_units.append(lost)
            avg_inventory.append(np.mean(inv_levels))
            total_demand_list.append(total_demand)
            orders_list.append(orders_count)

            daily_demand_matrix.append(daily_demands.tolist())
            if return_full_distribution and sim < 5:
                preview_inventory_paths.append(inv_levels.copy())

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
            "sample_path": {
                "inventory_path": sample_inventory_path,
                "demand_path": sample_demand_path,
                "stockout_days": sample_stockout_days,
                "reorder_days": sample_reorder_days,
                "arrival_days": sample_arrival_days
            },
            "monte_carlo_preview": {
                "total_demand": total_demand_list if return_full_distribution else [],
                "fill_rate": fill_rates if return_full_distribution else [],
                "lost_units": lost_units if return_full_distribution else [],
                "stockout_days": stockout_days if return_full_distribution else [],
                "inventory_paths": preview_inventory_paths if return_full_distribution else []
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

            demand = max(0.0, float(daily_demand[day]))
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
