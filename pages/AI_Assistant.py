import re
import streamlit as st
import anthropic
import pandas as pd

from backend.mcp_tools import event_window_shock_tool, portfolio_event_window_shock_tool
from backend.build_base_df import build_df_base
from backend.forecast_agent import ForecastAgent
from backend.optimization_agent import OptimizationAgent
from backend.simulation_agent import SimulationAgent
from backend.app_context import get_app_context

st.set_page_config(
    page_title="AI Inventory Assistant",
    layout="wide"
)

st.title("AI Inventory Strategy Assistant")
context = get_app_context()

client = anthropic.Anthropic(api_key="")
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a managerial inventory advisor for business stakeholders. "
    "Write in clear, everyday language that any non-technical person can understand. "
    "Be detailed enough to explain what is happening, why it matters, and what action makes sense, "
    "but do not overload the user with jargon or unnecessary detail. "
    "Prefer short, plain-English explanations over technical wording. "
    "Focus on business impact, trade-offs, risks, and practical next steps. "
    "Only use the provided structured data. Never invent numbers. "
    "Never assume demand increased unless explicitly stated. "
    "If data is missing, respond: 'That information is not available in the current results.' "
    "Keep responses to about 4 to 7 sentences unless the user asks for more detail. "
    "Use portfolio data only if the question relates to portfolio. "
    "Use single SKU data only if the question relates to a single SKU. "
    "Use shock data only if the question relates to shock. "
    "When discussing shocks, use the 'Demand shock increase (%)' value for the shock size, "
    "and do not confuse it with cost or fill-rate change percentages. "
    "When referencing percentages, copy the exact values shown in structured data. "
    "If appropriate, explain numbers in simple business terms, for example whether service stayed strong, "
    "cost rose meaningfully, or more inventory would be needed."
)

@st.cache_resource
def load_sim_agent():
    return SimulationAgent()

@st.cache_resource
def load_opt_agent(_sim_agent):
    return OptimizationAgent(simulation_agent=_sim_agent, order_cost=150)

@st.cache_data
def load_df():
    return build_df_base(store_ids=("CA_1", "TX_1", "WI_1"))

@st.cache_resource
def load_forecast_model(df):
    return ForecastAgent(model_dir="models", df_base=df)

sim_agent = load_sim_agent()
opt_agent = load_opt_agent(sim_agent)
df_base = load_df()
forecast_agent = load_forecast_model(df_base)


def is_shock_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in ("shock", "increase", "surge", "demand")
    ) or "%" in lowered


def extract_shock_percentage(text: str):
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value = value / 100
    return value


def build_structured_context(baseline_context, portfolio_context, shock_analysis):
    sections = []

    if baseline_context and baseline_context.get("policies"):
        policy = baseline_context.get("policies", [{}])[0] or {}
        sim_metrics = policy.get("simulation_metrics", {}) or {}
        inputs = baseline_context.get("inputs", {}) or {}
        assumptions = baseline_context.get("assumptions", {}) or {}

        sku_name = inputs.get("item_id") or inputs.get("sku") or "Unknown SKU"
        s_value = float(policy.get("s", 0) or 0)
        q_value = float(policy.get("Q", 0) or 0)
        target_service_level = float(assumptions.get("service_level_target", 0) or 0)
        fill_rate = float(policy.get("fill_rate", 0) or 0)
        stockout_prob = float(sim_metrics.get("stockout_probability", 0) or 0)
        avg_inventory = float(sim_metrics.get("avg_inventory", 0) or 0)
        expected_lost = float(sim_metrics.get("lost_units", 0) or 0)
        orders_per_year = float(sim_metrics.get("orders_per_year", sim_metrics.get("orders_placed", 0)) or 0)
        holding_cost = float(policy.get("holding_cost", 0) or 0)
        ordering_cost = float(policy.get("ordering_cost", 0) or 0)
        stockout_cost = float(policy.get("stockout_cost", 0) or 0)
        total_cost = float(policy.get("total_cost", 0) or 0)
        working_capital = float(policy.get("investment", 0) or 0)
        avg_price = float(
            policy.get("avg_price", inputs.get("avg_price", assumptions.get("avg_price", 0))) or 0
        )
        annual_procurement = q_value * avg_price * orders_per_year

        sections.append(
            "\n".join(
                [
                    "Single SKU Results:",
                    f"SKU: {sku_name}",
                    f"s: {s_value:.0f}",
                    f"Q: {q_value:.0f}",
                    f"Target service level: {target_service_level:.4f}",
                    f"Fill rate: {fill_rate:.4f}",
                    f"Stockout probability: {stockout_prob:.4f}",
                    f"Average inventory: {avg_inventory:.2f}",
                    f"Expected lost units: {expected_lost:.2f}",
                    f"Orders per year: {orders_per_year:.2f}",
                    f"Holding cost: {holding_cost:.2f}",
                    f"Ordering cost: {ordering_cost:.2f}",
                    f"Stockout cost: {stockout_cost:.2f}",
                    f"Total operating cost: {total_cost:.2f}",
                    f"Working capital: {working_capital:.2f}",
                    f"Annual procurement spend: {annual_procurement:.2f}",
                ]
            )
        )

    if portfolio_context:
        portfolio_meta = st.session_state.get("portfolio_meta", {}) or {}
        budget = float(portfolio_meta.get("budget", 0) or 0)
        total_capital = sum((policy.get("investment", 0) or 0) for policy in portfolio_context.values())
        total_cost = sum((policy.get("cost", 0) or 0) for policy in portfolio_context.values())
        total_procurement = sum((policy.get("procurement_spend", 0) or 0) for policy in portfolio_context.values())
        capital_utilization = (total_capital / budget * 100) if budget else 0

        sku_lines = []
        for sku, policy in portfolio_context.items():
            policy = policy or {}
            sim_metrics = policy.get("simulation_metrics", {}) or {}
            sku_capital = float(policy.get("investment", 0) or 0)
            sku_cost = float(policy.get("cost", 0) or 0)
            capital_share = (sku_capital / total_capital * 100) if total_capital else 0
            cost_share = (sku_cost / total_cost * 100) if total_cost else 0

            sku_lines.append(
                "\n".join(
                    [
                        f"SKU: {sku}",
                        f"s: {float(policy.get('s', 0) or 0):.0f}",
                        f"Q: {float(policy.get('Q', 0) or 0):.0f}",
                        f"Fill rate: {float(policy.get('fill_rate', 0) or 0):.4f}",
                        f"Stockout probability: {float(sim_metrics.get('stockout_probability', 0) or 0):.4f}",
                        f"Average inventory: {float(sim_metrics.get('avg_inventory', 0) or 0):.2f}",
                        f"Expected lost units: {float(sim_metrics.get('lost_units', 0) or 0):.2f}",
                        f"Holding cost: {float(policy.get('holding_cost', 0) or 0):.2f}",
                        f"Ordering cost: {float(policy.get('ordering_cost', 0) or 0):.2f}",
                        f"Working capital: {sku_capital:.2f}",
                        f"Operating cost: {sku_cost:.2f}",
                        f"Procurement spend: {float(policy.get('procurement_spend', 0) or 0):.2f}",
                        f"Capital share (%): {capital_share:.2f}",
                        f"Cost share (%): {cost_share:.2f}",
                    ]
                )
            )

        sections.append(
            "\n".join(
                [
                    "Portfolio Summary:",
                    f"Total SKUs: {len(portfolio_context)}",
                    f"Total working capital: {total_capital:.2f}",
                    f"Total operating cost: {total_cost:.2f}",
                    f"Total procurement spend: {total_procurement:.2f}",
                    f"Budget: {budget:.2f}",
                    f"Capital utilization (%): {capital_utilization:.2f}",
                    "Per-SKU details:",
                    *sku_lines,
                ]
            )
        )

    if shock_analysis:
        baseline = shock_analysis.get("baseline_policy", {}) or {}
        shock = shock_analysis.get("shock_optimal_policy", {}) or {}
        shock_percentage = float(shock_analysis.get("shock_percentage", st.session_state.get("last_shock_percentage", 0)) or 0)
        baseline_cost = float(baseline.get("total_cost", 0) or 0)
        shock_cost = float(shock.get("total_cost", 0) or 0)
        baseline_fill = float(baseline.get("fill_rate", 0) or 0)
        shock_fill = float(shock.get("fill_rate", 0) or 0)
        cost_change = ((shock_cost - baseline_cost) / baseline_cost * 100) if baseline_cost else 0
        fill_change = ((shock_fill - baseline_fill) / baseline_fill * 100) if baseline_fill else 0

        if shock_analysis.get("analysis_type") == "portfolio" or baseline.get("scope") == "portfolio":
            shocked = shock_analysis.get("shocked_policy", {}) or {}
            sections.append(
                "\n".join(
                    [
                        "Shock Analysis:",
                        "Scope: Portfolio",
                        f"Demand shock increase (%): {shock_percentage*100:.2f}",
                        f"Baseline combined s: {baseline.get('s', '0')}",
                        f"Baseline combined Q: {baseline.get('Q', '0')}",
                        f"Baseline portfolio fill rate: {baseline_fill:.4f}",
                        f"Baseline portfolio total cost: {baseline_cost:.2f}",
                        f"Baseline working capital: {float(baseline.get('working_capital', 0) or 0):.2f}",
                        f"Shocked combined s: {shocked.get('s', baseline.get('s', '0'))}",
                        f"Shocked combined Q: {shocked.get('Q', baseline.get('Q', '0'))}",
                        f"Shocked portfolio fill rate: {float(shocked.get('fill_rate', 0) or 0):.4f}",
                        f"Shocked portfolio total cost: {float(shocked.get('total_cost', 0) or 0):.2f}",
                        f"Re-optimized combined s: {shock.get('s', '0')}",
                        f"Re-optimized combined Q: {shock.get('Q', '0')}",
                        f"Re-optimized portfolio fill rate: {shock_fill:.4f}",
                        f"Re-optimized portfolio total cost: {shock_cost:.2f}",
                        f"Cost change vs baseline (%): {cost_change:.2f}",
                        f"Fill rate change vs baseline (%): {fill_change:.2f}",
                        f"Risk level: {shock.get('risk_level', 'Unknown')}",
                    ]
                )
            )
        else:
            sections.append(
                "\n".join(
                    [
                        "Shock Analysis:",
                        "Scope: Single SKU",
                        f"Demand shock increase (%): {shock_percentage*100:.2f}",
                        f"Baseline s: {float(baseline.get('s', 0) or 0):.0f}",
                        f"Baseline Q: {float(baseline.get('Q', 0) or 0):.0f}",
                        f"Baseline fill rate: {baseline_fill:.4f}",
                        f"Baseline total cost: {baseline_cost:.2f}",
                        f"Shock s: {float(shock.get('s', 0) or 0):.0f}",
                        f"Shock Q: {float(shock.get('Q', 0) or 0):.0f}",
                        f"Shock fill rate: {shock_fill:.4f}",
                        f"Shock total cost: {shock_cost:.2f}",
                        f"Cost change vs baseline (%): {cost_change:.2f}",
                        f"Fill rate change vs baseline (%): {fill_change:.2f}",
                        f"Risk level: {shock.get('risk_level', 'Unknown')}",
                    ]
                )
            )

    if not sections:
        return "No optimization results available yet."

    return "\n\n".join(sections)


def render_shock_table(shock_result):
    baseline = shock_result["baseline_policy"]
    shocked = shock_result.get("shocked_policy", {})
    shock = shock_result["shock_optimal_policy"]
    is_portfolio = shock_result.get("analysis_type") == "portfolio" or baseline.get("scope") == "portfolio"

    if is_portfolio:
        comparison_df = pd.DataFrame(
            [
                {
                    "Policy": "Baseline",
                    "s": baseline.get("s", "0"),
                    "Q": baseline.get("Q", "0"),
                    "SKUs": baseline.get("sku_count", 0),
                    "Working Capital": baseline.get("working_capital", 0),
                    "Fill Rate": baseline.get("fill_rate", 0),
                    "Total Cost": baseline.get("total_cost", 0),
                    "Risk Level": baseline.get("risk_level", "")
                },
                {
                    "Policy": "Shocked",
                    "s": shocked.get("s", baseline.get("s", "0")),
                    "Q": shocked.get("Q", baseline.get("Q", "0")),
                    "SKUs": shocked.get("sku_count", baseline.get("sku_count", 0)),
                    "Working Capital": shocked.get("working_capital", 0),
                    "Fill Rate": shocked.get("fill_rate", 0),
                    "Total Cost": shocked.get("total_cost", 0),
                    "Risk Level": shocked.get("risk_level", "")
                },
                {
                    "Policy": "Shock-Optimized",
                    "s": shock.get("s", "0"),
                    "Q": shock.get("Q", "0"),
                    "SKUs": shock.get("sku_count", 0),
                    "Working Capital": shock.get("working_capital", 0),
                    "Fill Rate": shock.get("fill_rate", 0),
                    "Total Cost": shock.get("total_cost", 0),
                    "Risk Level": shock.get("risk_level", "")
                }
            ]
        )
    else:
        comparison_df = pd.DataFrame(
            [
                {
                    "Policy": "Baseline",
                    "s": baseline["s"],
                    "Q": baseline["Q"],
                    "Fill Rate": baseline["fill_rate"],
                    "Total Cost": baseline["total_cost"],
                    "Risk Level": baseline.get("risk_level", "")
                },
                {
                    "Policy": "Shocked",
                    "s": shocked.get("s", baseline["s"]),
                    "Q": shocked.get("Q", baseline["Q"]),
                    "Fill Rate": shocked.get("fill_rate", 0),
                    "Total Cost": shocked.get("total_cost", 0),
                    "Risk Level": shocked.get("risk_level", "")
                },
                {
                    "Policy": "Shock-Optimized",
                    "s": shock["s"],
                    "Q": shock["Q"],
                    "Fill Rate": shock["fill_rate"],
                    "Total Cost": shock["total_cost"],
                    "Risk Level": shock.get("risk_level", "")
                }
            ]
        )

    def style_risk_level(value):
        color_map = {
            "Low Risk": "background-color: rgba(0, 200, 0, 0.15); color: #36d060; font-weight: 600;",
            "Moderate Risk": "background-color: rgba(255, 165, 0, 0.15); color: #ffb347; font-weight: 600;",
            "High Risk": "background-color: rgba(255, 99, 71, 0.18); color: #ff6b6b; font-weight: 600;",
        }
        return color_map.get(value, "")

    def style_change(row):
        if row["Policy"] == "Baseline":
            return [""] * len(row)

        if row["Policy"] == "Shocked":
            reference_row = comparison_df.loc[comparison_df["Policy"] == "Baseline"].iloc[0]
        else:
            shocked_rows = comparison_df.loc[comparison_df["Policy"] == "Shocked"]
            if not shocked_rows.empty:
                reference_row = shocked_rows.iloc[0]
            else:
                reference_row = comparison_df.loc[comparison_df["Policy"] == "Baseline"].iloc[0]

        styles = []
        for col in row.index:
            if col == "Fill Rate":
                styles.append(
                    "color: #36d060; font-weight: 600;"
                    if row[col] >= reference_row[col]
                    else "color: #ff6b6b; font-weight: 600;"
                )
            elif col == "Total Cost":
                styles.append(
                    "color: #36d060; font-weight: 600;"
                    if row[col] <= reference_row[col]
                    else "color: #ff6b6b; font-weight: 600;"
                )
            elif col == "Working Capital":
                styles.append(
                    "color: #36d060; font-weight: 600;"
                    if row[col] <= reference_row[col]
                    else "color: #ff6b6b; font-weight: 600;"
                )
            else:
                styles.append("")
        return styles

    styled = (
        comparison_df.style
        .format({
            "Fill Rate": "{:.2%}",
            "Total Cost": "${:,.2f}",
            "Working Capital": "${:,.2f}"
        })
        .applymap(style_risk_level, subset=["Risk Level"])
        .apply(style_change, axis=1)
    )

    st.caption(
        "Comparison of baseline policy performance under normal demand, shocked demand, and after policy re-optimization."
    )
    if is_portfolio:
        st.caption("For portfolio shocks, `s` and `Q` show a combined policy summary as average values with the min-max range across SKUs when policies differ.")
    st.dataframe(styled, use_container_width=True)


def render_risk_badge(label: str):
    color_map = {
        "Low Risk": "background-color: rgba(0, 200, 0, 0.15); color: #36d060;",
        "Moderate Risk": "background-color: rgba(255, 165, 0, 0.15); color: #ffb347;",
        "High Risk": "background-color: rgba(255, 99, 71, 0.18); color: #ff6b6b;",
    }
    style = color_map.get(label, "background-color: rgba(148, 163, 184, 0.15); color: #e2e8f0;")
    st.markdown(
        f"<span style='{style} padding: 4px 8px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;'>"
        f"{label}</span>",
        unsafe_allow_html=True
    )


def render_shock_metrics_row(shock_result):
    baseline = shock_result.get("baseline_policy", {}) or {}
    shock = shock_result.get("shock_optimal_policy", {}) or {}
    is_portfolio = shock_result.get("analysis_type") == "portfolio" or baseline.get("scope") == "portfolio"

    cols = st.columns(5)
    cols[0].metric("Baseline Fill Rate", f"{float(baseline.get('fill_rate', 0) or 0):.2%}")
    cols[1].metric("Shock Fill Rate", f"{float(shock.get('fill_rate', 0) or 0):.2%}")
    cols[2].metric("Baseline Cost", f"${float(baseline.get('total_cost', 0) or 0):,.2f}")
    metric_label = "Shock Cost" if not is_portfolio else "Re-Optimized Cost"
    cols[3].metric(metric_label, f"${float(shock.get('total_cost', 0) or 0):,.2f}")
    with cols[4]:
        caption = "Risk Level" if not is_portfolio else "Portfolio Risk"
        st.caption(caption)
        render_risk_badge(shock.get("risk_level", "Unknown"))


def use_portfolio_shock_mode(user_prompt, baseline_context, portfolio_context):
    """
    Decide whether a shock request should run against the portfolio context.
    """
    lowered = user_prompt.lower()
    portfolio_keywords = (
        "portfolio",
        "multi sku",
        "multiple sku",
        "all sku",
        "all skus",
        "budget",
    )
    if portfolio_context and baseline_context is None:
        return True
    if portfolio_context and any(keyword in lowered for keyword in portfolio_keywords):
        return True
    return False


def render_shock_recommendations():
    st.markdown("**Recommended Actions**")
    st.markdown(
        "- Increase the reorder point to maintain service levels under higher demand.\n"
        "- Monitor working capital due to increased inventory holding.\n"
        "- Validate supplier capacity for sustained demand surges.\n"
        "- Consider testing additional scenarios (e.g., 50%, 100%, 150%)."
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = context.get("chat_history", [])

if "shock_analysis" not in st.session_state:
    st.session_state.shock_analysis = None
if "last_shock_prompt" not in st.session_state:
    st.session_state.last_shock_prompt = None
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = None
if "show_shock_table" not in st.session_state:
    st.session_state.show_shock_table = False
if "last_shock_percentage" not in st.session_state:
    st.session_state.last_shock_percentage = 0.0
if "shock_history" not in st.session_state:
    st.session_state.shock_history = []

baseline_context = st.session_state.get("baseline_context") or context.get("single_sku_results")
portfolio_context = st.session_state.get("portfolio_context") or context.get("portfolio_results")
portfolio_meta = context.get("portfolio_meta", {}) or {}

# Render chat history
for message in st.session_state.chat_history:
    avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

st.markdown("---")

user_prompt = st.chat_input("Ask something about your inventory strategy...")

if user_prompt:
    st.session_state.last_user_prompt = user_prompt
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    context["chat_history"] = st.session_state.chat_history

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_prompt)

    response_text = ""

    shock_intent = is_shock_intent(user_prompt)
    shock_percentage = extract_shock_percentage(user_prompt)

    if shock_intent and shock_percentage is None:
        if st.session_state.shock_analysis is None:
            response_text = (
                "I can run a shock analysis. Please specify the demand increase percentage "
                "(e.g., 20%, 50%, or 100%)."
            )
        else:
            structured_context = build_structured_context(
                baseline_context,
                portfolio_context,
                st.session_state.shock_analysis
            )
            conversation_text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history]
            )

            prompt = (
                "Important: If you mention shock size, use the exact 'Demand shock increase (%)' value "
                "from the structured data. Do not substitute cost or fill-rate change percentages.\n\n"
                "Structured Data:\n"
                f"{structured_context}\n\n"
                "Conversation:\n"
                f"{conversation_text}\n"
            )

            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            response_text = response.content[0].text
    elif shock_intent and shock_percentage is not None:
        use_portfolio_mode = use_portfolio_shock_mode(
            user_prompt,
            baseline_context,
            portfolio_context
        )

        if baseline_context is None and portfolio_context is None:
            response_text = (
                "Please run a single SKU or portfolio optimization first so I can analyze "
                "your inventory strategy."
            )
        else:
            if st.session_state.last_shock_prompt == user_prompt and st.session_state.shock_analysis is not None:
                response_text = (
                    "Using the existing shock analysis from your last request. "
                    "Let me know if you want a different percentage."
                )
            else:
                if use_portfolio_mode:
                    result = portfolio_event_window_shock_tool(
                        portfolio_context=portfolio_context,
                        portfolio_meta=portfolio_meta,
                        sim_agent=sim_agent,
                        opt_agent=opt_agent,
                        df_base=df_base,
                        forecast_agent=forecast_agent,
                        shock_percentage=shock_percentage
                    )
                else:
                    result = event_window_shock_tool(
                        baseline_context=baseline_context,
                        sim_agent=sim_agent,
                        opt_agent=opt_agent,
                        df_base=df_base,
                        forecast_agent=forecast_agent,
                        shock_percentage=shock_percentage
                    )

                if "error" in result:
                    response_text = result["error"]
                else:
                    st.session_state.shock_analysis = result
                    st.session_state.last_shock_prompt = user_prompt
                    st.session_state.last_shock_percentage = shock_percentage
                    st.session_state.show_shock_table = True
                    st.session_state.shock_history.append(
                        {
                            "percentage": shock_percentage,
                            "result": result
                        }
                    )
                    if use_portfolio_mode:
                        response_text = (
                            f"I ran a {shock_percentage*100:.0f}% demand shock on the portfolio. "
                            "This shows how the current portfolio performs under higher demand and what a re-optimized "
                            "portfolio would look like under the same budget."
                        )
                    else:
                        response_text = (
                            f"I ran a {shock_percentage*100:.0f}% demand shock. "
                            "The policy shifts to buffer higher demand while balancing cost and service. "
                            "Review the new reorder point and order quantity to ensure the trade-off fits your risk tolerance."
                        )
    else:
        if baseline_context is None and portfolio_context is None:
            response_text = (
                "Please run a single SKU or portfolio optimization first so I can analyze "
                "your inventory strategy."
            )
        else:
            structured_context = build_structured_context(
                baseline_context,
                portfolio_context,
                st.session_state.shock_analysis
            )
            conversation_text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history]
            )

            prompt = (
                "Structured Data:\n"
                f"{structured_context}\n\n"
                "Conversation:\n"
                f"{conversation_text}\n"
            )

            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            response_text = response.content[0].text

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    context["chat_history"] = st.session_state.chat_history

st.markdown("---")

if st.session_state.shock_history:
    for entry in st.session_state.shock_history:
        shock_pct = float(entry.get("percentage", 0) or 0)
        shock_result = entry.get("result")
        if not shock_result:
            continue
        is_portfolio = (
            shock_result.get("analysis_type") == "portfolio"
            or shock_result.get("baseline_policy", {}).get("scope") == "portfolio"
        )
        st.markdown("---")
        st.markdown(
            f"### ⚡ {'Portfolio ' if is_portfolio else ''}Shock Analysis: {shock_pct*100:.0f}% Demand Increase"
        )
        st.markdown("#### 📊 Metrics")
        render_shock_metrics_row(shock_result)
        st.markdown("---")
        render_shock_table(shock_result)
        st.markdown("---")
        st.markdown("#### ✅ Recommended Actions")
        render_shock_recommendations()
