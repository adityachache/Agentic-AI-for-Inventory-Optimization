import re
import streamlit as st
from google import genai
import pandas as pd

from backend.mcp_tools import event_window_shock_tool
from backend.build_base_df import build_df_base
from backend.forecast_agent import ForecastAgent
from backend.optimization_agent import OptimizationAgent
from backend.simulation_agent import SimulationAgent
from backend.app_context import get_app_context

st.set_page_config(
    page_title="AI Inventory Assistant",
    layout="centered"
)

st.title("AI Inventory Strategy Assistant")
context = get_app_context()

llm_client = genai.Client(api_key="AIzaSyAugNWSg-ABLT-X5UjJURA_hjWVykyA4VY")
llm_model = "gemini-3-flash-preview"

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


def is_shock_request(text: str) -> bool:
    lowered = text.lower()
    has_number = re.search(r"\d+(?:\.\d+)?", lowered) is not None
    has_keyword = any(
        key in lowered
        for key in ("shock", "increase", "surge", "demand")
    ) or "%" in lowered
    return has_number and has_keyword


def build_structured_context(baseline_context, portfolio_context, shock_analysis):
    sections = []

    if baseline_context and baseline_context.get("policies"):
        policy = baseline_context["policies"][0]
        sim_metrics = policy.get("simulation_metrics", {})
        item_id = baseline_context.get("inputs", {}).get("item_id", "Unknown SKU")
        sections.append(
            "\n".join(
                [
                    "Single SKU Results:",
                    f"SKU: {item_id}",
                    f"Reorder point: {policy.get('s')}",
                    f"Order quantity: {policy.get('Q')}",
                    f"Fill rate: {policy.get('fill_rate', 0):.4f}",
                    f"Total cost: {policy.get('total_cost', 0):.2f}",
                    f"Average inventory: {sim_metrics.get('avg_inventory', 0):.2f}",
                    f"Expected lost units: {sim_metrics.get('lost_units', 0):.2f}",
                ]
            )
        )

    if portfolio_context:
        total_capital = sum(policy.get("investment", 0) for policy in portfolio_context.values())
        total_cost = sum(policy.get("cost", 0) for policy in portfolio_context.values())
        total_procurement = sum(policy.get("procurement_spend", 0) for policy in portfolio_context.values())
        sku_lines = [
            f"{sku}: s={policy.get('s')}, Q={policy.get('Q')}, "
            f"working_capital={policy.get('investment', 0):.2f}, "
            f"operating_cost={policy.get('cost', 0):.2f}, "
            f"procurement_spend={policy.get('procurement_spend', 0):.2f}"
            for sku, policy in portfolio_context.items()
        ]
        sections.append(
            "\n".join(
                [
                    "Portfolio Summary:",
                    f"SKUs: {len(portfolio_context)}",
                    f"Total working capital: {total_capital:.2f}",
                    f"Total operating cost: {total_cost:.2f}",
                    f"Total procurement spend: {total_procurement:.2f}",
                    "Per-SKU details:",
                    *sku_lines,
                ]
            )
        )

    if shock_analysis:
        baseline = shock_analysis.get("baseline_policy", {})
        shock = shock_analysis.get("shock_optimal_policy", {})
        sections.append(
            "\n".join(
                [
                    "Shock Analysis Results:",
                    f"Baseline s: {baseline.get('s')}",
                    f"Baseline Q: {baseline.get('Q')}",
                    f"Baseline fill rate: {baseline.get('fill_rate', 0):.4f}",
                    f"Baseline total cost: {baseline.get('total_cost', 0):.2f}",
                    f"Shock s: {shock.get('s')}",
                    f"Shock Q: {shock.get('Q')}",
                    f"Shock fill rate: {shock.get('fill_rate', 0):.4f}",
                    f"Shock total cost: {shock.get('total_cost', 0):.2f}",
                    f"Shock risk level: {shock.get('risk_level', 'Unknown')}",
                ]
            )
        )

    if not sections:
        return "No optimization results available yet."

    return "\n\n".join(sections)


def render_shock_table(shock_result):
    baseline = shock_result["baseline_policy"]
    shock = shock_result["shock_optimal_policy"]

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
        if row["Policy"] != "Shock-Optimized":
            return [""] * len(row)
        styles = []
        for col in row.index:
            if col == "Fill Rate":
                styles.append(
                    "color: #36d060; font-weight: 600;"
                    if row[col] > comparison_df.loc[0, col]
                    else "color: #ff6b6b; font-weight: 600;"
                )
            elif col == "Total Cost":
                styles.append(
                    "color: #36d060; font-weight: 600;"
                    if row[col] < comparison_df.loc[0, col]
                    else "color: #ff6b6b; font-weight: 600;"
                )
            else:
                styles.append("")
        return styles

    styled = (
        comparison_df.style
        .format({"Fill Rate": "{:.2%}", "Total Cost": "${:,.2f}"})
        .applymap(style_risk_level, subset=["Risk Level"])
        .apply(style_change, axis=1)
    )

    st.dataframe(styled, use_container_width=True)


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

baseline_context = st.session_state.get("baseline_context") or context.get("single_sku_results")
portfolio_context = st.session_state.get("portfolio_context") or context.get("portfolio_results")

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

    if is_shock_request(user_prompt):
        if baseline_context is None:
            response_text = (
                "I can run a shock analysis once a single SKU optimization is available. "
                "Run a single SKU optimization first, then ask again."
            )
        else:
            if st.session_state.last_shock_prompt == user_prompt and st.session_state.shock_analysis is not None:
                response_text = (
                    "Using the existing shock analysis from your last request. "
                    "Let me know if you want a different percentage."
                )
            else:
                match = re.search(r"(\d+(?:\.\d+)?)", user_prompt)
                shock_percentage = float(match.group(1)) if match else 0.2
                if shock_percentage > 1:
                    shock_percentage = shock_percentage / 100

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
                    st.session_state.show_shock_table = True
                    response_text = (
                        f"I ran a {shock_percentage*100:.0f}% demand shock. "
                        "The policy shifts to buffer higher demand while balancing cost and service. "
                        "Review the new reorder point and order quantity to ensure the trade-off fits your risk tolerance."
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

        prompt = f"""
You are a practical inventory manager advising a retail business.
Only use the provided structured data for any facts or numbers.
Do not assume demand increased unless explicitly stated.
If data is missing, say you do not have it.
Keep the response under 6 sentences. Use plain conversational business language.

Structured Data:
{structured_context}

Conversation:
{conversation_text}
"""

        llm_response = llm_client.models.generate_content(
            model=llm_model,
            contents=prompt
        )
        response_text = llm_response.text

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    context["chat_history"] = st.session_state.chat_history

st.markdown("---")

if st.session_state.shock_analysis is not None and st.session_state.show_shock_table:
    render_shock_table(st.session_state.shock_analysis)
