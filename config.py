# config.py
import argparse
import os # os is still useful for path joining etc.

# --- Default Configuration Values ---
# File Paths
DEFAULT_STRUCTURED_CASE_PATH = './data/tcmd_eval.json'
DEFAULT_OUTPUT_JSON_PATH = './results/evaluation_results.json' # Used for both modes
DEFAULT_PROMPTS_DIR = './prompts'

# --- Default prompt file names ---
DEFAULT_PATIENT_PROMPT_FILE = "patient.txt"
DEFAULT_DOCTOR_PROMPT_FILE = "doctor.txt"
DEFAULT_ASSISTANT_PROMPT_FILE = "assistant.txt"
DEFAULT_ROUTER_PROMPT_FILE = "router.txt"
DEFAULT_EXPERT_PROMPT_FILE = "expert.txt"
DEFAULT_ONE_STEP_DOCTOR_PROMPT_FILE = "one_step_doctor.txt"
DEFAULT_ONE_STEP_EXPERT_PROMPT_FILE = "one_step_expert.txt"

# Simulation API Config (Used only in multi-turn mode)
DEFAULT_SIM_API_BASE_URL = "https://api.openai.com/v1" # Or your proxy URL
DEFAULT_SIM_API_KEY = "YOUR_SIM_API_KEY_HERE" # **Replace with your key or leave empty**
DEFAULT_SIM_MODEL_NAME = "gpt-4o-mini"

# Expert API Config (Used in both modes)
DEFAULT_EXPERT_API_BASE_URL = DEFAULT_SIM_API_BASE_URL # Often same as sim URL
DEFAULT_EXPERT_API_KEY = DEFAULT_SIM_API_KEY # Often same as sim key
DEFAULT_EXPERT_MODEL_NAME = "gpt-4o" # Or gpt-4o-2024-08-06 etc.

# Test Model API Config (Used in both modes)
DEFAULT_TEST_API_BASE_URL = "http://localhost:5000/v1" # Default VLLM endpoint
DEFAULT_TEST_API_KEY = "EMPTY" # Default for many local setups
DEFAULT_TEST_MODEL_NAME = "LLM_API" # Default served model name

# Dialogue Simulation Parameters (Used only in multi-turn mode)
DEFAULT_MAX_DIALOGUE_TURNS = 10 # Default max turns before forcing diagnosis

# --- Execution Mode ---
DEFAULT_MODE = 'multi-turn' # Default mode is the original simulation

# --- Configuration Loading Function ---

def load_config():
    """
    Loads configuration by defining defaults in this file
    and allowing overrides via command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run medical evaluation in different modes.')

    # --- Execution Mode ---
    parser.add_argument('--mode', type=str, choices=['multi-turn', 'one-step'],
                        default=DEFAULT_MODE,
                        help=f'Execution mode: multi-turn simulation or one-step diagnosis (default: {DEFAULT_MODE})')

    # File Paths
    parser.add_argument('--structed_case_path', type=str,
                        default=DEFAULT_STRUCTURED_CASE_PATH,
                        help=f'Path to the structured medical cases JSON file (default: {DEFAULT_STRUCTURED_CASE_PATH})')
    parser.add_argument('--output_json_path', type=str,
                        default=DEFAULT_OUTPUT_JSON_PATH,
                        help=f'Path to save the evaluation results JSON file (default: {DEFAULT_OUTPUT_JSON_PATH})')
    parser.add_argument('--prompts_dir', type=str,
                        default=DEFAULT_PROMPTS_DIR,
                        help=f'Directory containing prompt template files (default: {DEFAULT_PROMPTS_DIR})')

    # Simulation API Config (Only needed for multi-turn)
    parser.add_argument('--sim_base_url', type=str,
                        default=DEFAULT_SIM_API_BASE_URL,
                        help=f'Base URL for the simulation LLM API (multi-turn mode) (default: {DEFAULT_SIM_API_BASE_URL})')
    parser.add_argument('--sim_api_key', type=str,
                        default=DEFAULT_SIM_API_KEY,
                        help='API Key for the simulation LLM API (multi-turn mode) (default: set in config.py)')
    parser.add_argument('--sim_model_name', type=str,
                        default=DEFAULT_SIM_MODEL_NAME,
                        help=f'Model name for the simulation LLM (multi-turn mode) (default: {DEFAULT_SIM_MODEL_NAME})')

    # Expert API Config
    parser.add_argument('--expert_base_url', type=str,
                        default=DEFAULT_EXPERT_API_BASE_URL,
                        help=f'Base URL for the expert evaluator LLM API (default: {DEFAULT_EXPERT_API_BASE_URL})')
    parser.add_argument('--expert_api_key', type=str,
                        default=DEFAULT_EXPERT_API_KEY,
                        help='API Key for the expert evaluator LLM API (default: set in config.py)')
    parser.add_argument('--expert_model_name', type=str,
                        default=DEFAULT_EXPERT_MODEL_NAME,
                        help=f'Model name for the expert evaluator LLM (default: {DEFAULT_EXPERT_MODEL_NAME})')

    # Test Model API Config
    parser.add_argument('--test_base_url', type=str,
                        default=DEFAULT_TEST_API_BASE_URL,
                        help=f'Base URL for the model under test API (default: {DEFAULT_TEST_API_BASE_URL})')
    parser.add_argument('--test_api_key', type=str,
                        default=DEFAULT_TEST_API_KEY,
                        help=f'API Key for the model under test API (default: {DEFAULT_TEST_API_KEY})')
    parser.add_argument('--test_model_name', type=str,
                        default=DEFAULT_TEST_MODEL_NAME,
                        help=f'Model name for the model under test (default: {DEFAULT_TEST_MODEL_NAME})')

    # Dialogue Parameters (Only needed for multi-turn)
    parser.add_argument('--max_dialogue_turns', type=int,
                        default=DEFAULT_MAX_DIALOGUE_TURNS,
                        help=f'Maximum number of doctor turns in multi-turn mode (default: {DEFAULT_MAX_DIALOGUE_TURNS})')

    args = parser.parse_args()

    # --- Assign default prompt file names to config object ---
    # Makes them accessible via config.xxx_prompt_file later
    args.patient_prompt_file = DEFAULT_PATIENT_PROMPT_FILE
    args.doctor_prompt_file = DEFAULT_DOCTOR_PROMPT_FILE
    args.assistant_prompt_file = DEFAULT_ASSISTANT_PROMPT_FILE
    args.router_prompt_file = DEFAULT_ROUTER_PROMPT_FILE
    args.expert_prompt_file = DEFAULT_EXPERT_PROMPT_FILE
    args.one_step_doctor_prompt_file = DEFAULT_ONE_STEP_DOCTOR_PROMPT_FILE
    args.one_step_expert_prompt_file = DEFAULT_ONE_STEP_EXPERT_PROMPT_FILE


    # --- Validation ---
    # Only validate sim_api_key if in multi-turn mode
    if args.mode == 'multi-turn' and (not args.sim_api_key or args.sim_api_key == "YOUR_SIM_API_KEY_HERE"):
        print(f"Warning: Simulation API Key ('{args.sim_api_key}') seems unconfigured (needed for multi-turn mode). Check config.py or use --sim_api_key.")

    # Validate keys needed in both modes
    if not args.expert_api_key or args.expert_api_key == "YOUR_SIM_API_KEY_HERE": # Check expert key (uses sim key default)
         print(f"Warning: Expert API Key ('{args.expert_api_key}') seems unconfigured. Check config.py or use --expert_api_key.")

    if not args.test_base_url or not args.test_model_name:
         print(f"Warning: Test model URL ('{args.test_base_url}') or name ('{args.test_model_name}') is missing.")

    # Add more validation as needed

    return args