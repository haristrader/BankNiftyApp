# pages/06_AI_Console.py
import streamlit as st

st.title("🤖 AI Learning Console (Preview)")

st.write("""
- Tracks per-module performance over time
- Suggests weight changes (asks before applying)
- Logs: losing streaks, module drift, fib/trend accuracy
""")

st.info("Coming soon. Scores already flow to Dashboard; this page will manage weight suggestions.")
