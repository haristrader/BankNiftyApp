# supabase_client.py
from typing import Optional
import os

from supabase import create_client, Client

# Prefer Streamlit secrets for security (works locally & Streamlit Cloud)
SUPABASE_URL = os.getenv("SUPABASE_URL") or (
    # st.secrets compatible fallback (no hard dependency on streamlit here)
    None
)
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or None

# If environment vars absent, use your provided keys (you can remove later)
if not SUPABASE_URL:
    SUPABASE_URL = "https://gfdzavlnzlxnpgctihrm.supabase.co"
if not SUPABASE_KEY:
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdmZHphdmxuemx4bnBnY3RpaHJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA4MjI5MDIsImV4cCI6MjA3NjM5ODkwMn0.X_2d7CNgOWd8BWVnkfj5gFsmKxa7Pm2PfLs7gawJN4U"

_client: Optional[Client] = None

def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
