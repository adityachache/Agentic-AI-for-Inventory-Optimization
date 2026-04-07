import pyomo.environ as pyo


def solve_portfolio(candidate_data, budget):

    model = pyo.ConcreteModel()

    skus = list(candidate_data.keys())

    # Build index list
    index_set = []
    for sku in skus:
        for policy in candidate_data[sku]:
            index_set.append((sku, policy["policy_id"]))

    model.x = pyo.Var(index_set, domain=pyo.Binary)

    # Objective: minimize total cost
    def objective_rule(m):
        return sum(
            next(p["cost"] for p in candidate_data[sku]
                 if p["policy_id"] == pid)
            * m.x[(sku, pid)]
            for (sku, pid) in index_set
        )

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Budget constraint
    def budget_rule(m):
        return sum(
            next(p["investment"] for p in candidate_data[sku]
                 if p["policy_id"] == pid)
            * m.x[(sku, pid)]
            for (sku, pid) in index_set
        ) <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # One policy per SKU
    def one_policy_rule(m, sku):
        return sum(
            m.x[(sku, p["policy_id"])]
            for p in candidate_data[sku]
        ) == 1

    model.one_policy = pyo.Constraint(skus, rule=one_policy_rule)

    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model)

    # Check solver status
    if (results.solver.status != pyo.SolverStatus.ok or
        results.solver.termination_condition != pyo.TerminationCondition.optimal):
        return {"error": "Portfolio optimization infeasible under current budget."}

    selected = {}

    for sku in skus:
        for policy in candidate_data[sku]:
            pid = policy["policy_id"]
            if pyo.value(model.x[(sku, pid)]) == 1:
                selected[sku] = policy

    return selected