import litellm
import os
from dotenv import load_dotenv

# Load environment variables from .env file (especially POE_API_KEY).
# This allows users to set their POE_API_KEY in a .env file for local development.
# In production or containerized environments, environment variables should be set directly.
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("No .env file found or python-dotenv not installed. Relying on shell environment variables.")

# The PoeAdapter class is in poe_adapter.py and will be loaded by LiteLLM
# based on the configuration in litellm_config.yaml.

if __name__ == "__main__":
    print("Starting LiteLLM server with PoeAdapter support...")
    print("Ensure your POE_API_KEY is set in your environment (e.g., in a .env file).")
    print("Models and PoeAdapter registration are configured in 'litellm_config.yaml'.")

    # For debugging: Check if POE_API_KEY is accessible
    poe_api_key_env = os.environ.get("POE_API_KEY")
    if poe_api_key_env:
        print(f"POE_API_KEY found in environment (length: {len(poe_api_key_env)}).")
    else:
        print("Warning: POE_API_KEY not found in environment. Poe models will likely fail to initialize.")

    # Start the LiteLLM server using the configuration file.
    # LiteLLM will handle importing 'poe_adapter.PoeAdapter' as specified in the config.
    try:
        litellm.start_server(
            config_path="litellm_config.yaml",
            # host="0.0.0.0", # Default is 0.0.0.0, can be overridden by LITELLM_HOST env var or in config
            # port=8000,      # Default is 8000, can be overridden by LITELLM_PORT env var or in config
            # debug=True,     # Can be set via LITELLM_DEBUG env var or in config
            # num_workers=1,  # Default, adjust if needed
        )
    except Exception as e:
        print(f"Failed to start LiteLLM server: {e}")
        print("Please check your configuration, especially 'litellm_config.yaml' and POE_API_KEY.")

    # The server runs until manually stopped (e.g., Ctrl+C).
    # Code here will only execute after the server has shut down.
    print("LiteLLM server has shut down.")
