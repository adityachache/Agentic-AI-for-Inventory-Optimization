import streamlit as st


def get_app_context():
    if "app_context" not in st.session_state:
        st.session_state.app_context = {
            "single_sku_results": None,
            "portfolio_results": None,
            "monte_carlo_preview": None,
            "chat_history": []
        }
    return st.session_state.app_context
