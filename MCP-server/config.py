import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# This will find and load a `.env` file from your project root (final-yr-proj/.env)
# This is safe for production: it just won't find a .env file and will
# rely on real environment variables.
load_dotenv()

# --- Server Configuration ---
SERVER_NAME = os.getenv("SERVER_NAME", "AgentTools-MCP")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Tool-Specific API Keys & Config ---

# For Google Search (via SerpAPI)
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# For Wikipedia API (a custom user-agent is required by their ToS)
# Set this to something unique, like 'MyProject (myemail@example.com)'
WIKI_USER_AGENT = os.getenv("WIKI_USER_AGENT", "MCP-Server (contact@example.com)")


# --- Sanity Checks & Logging ---
# This helps you debug if your keys are loaded correctly on startup.

if SERPAPI_KEY:
    logger.info("SerpAPI key loaded successfully.")
else:
    logger.warning("SERPAPI_KEY not found in environment. Google Search tools will fail.")

if WIKI_USER_AGENT == "MCP-Server (contact@example.com)":
    logger.warning("Using default WIKI_USER_AGENT. Please set a custom one in your .env file.")
else:
    logger.info(f"Wikipedia User-Agent loaded: {WIKI_USER_AGENT}")