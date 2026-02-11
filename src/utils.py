# src/utils.py
import json
import os
import logging
import re
from typing import Any, Dict, List

# Use logging configured in main.py - get logger instance
logger = logging.getLogger(__name__)

def load_json(file_path: str) -> Any:
    """Loads JSON data from a file."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}. Returning empty list.")
        return [] # Default to empty list for results/data loading
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Basic check: Ensure it's a list if we typically expect one for results/cases
            # Modify this check based on actual expected format if needed
            # if not isinstance(data, list):
            #     logger.warning(f"File {file_path} does not contain a JSON list as might be expected.")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise # Re-raise after logging
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

def save_json(data: Any, file_path: str, indent: int = 4):
    """Saves data to a JSON file."""
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name: # Create directory only if path includes one
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        # Logging successful saves might be too verbose, use debug if needed
        # logger.debug(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        raise

def load_prompt(prompt_filename: str, prompts_dir: str) -> str:
    """Loads a prompt template from the specified prompts directory and filename."""
    # Ensure filename ends with .txt if not provided
    if not prompt_filename.endswith(".txt"):
        prompt_filename += ".txt"
    file_path = os.path.join(prompts_dir, prompt_filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {file_path}: {e}")
        raise

def parse_expert_evaluation(response_content: str) -> Dict[str, Any]:
    """
    Parses the expert evaluation JSON from the LLM response string.
    Handles potential ```json ... ``` blocks.
    """
    logger.debug(f"Attempting to parse expert evaluation from raw response: {response_content[:500]}...")
    json_string = None
    parsed_json = None
    try:
        # Try extracting from ```json ... ``` block first
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if match:
            json_string = match.group(1)
            logger.debug("Extracted JSON using ```json regex.")
        else:
            # Fallback: find first '{' and last '}'
            start_index = response_content.find('{')
            end_index = response_content.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_string = response_content[start_index : end_index + 1]
                logger.debug("Extracted JSON using find '{' and '}'.")
            else:
                logger.error("Could not find valid JSON structure ({...} or ```json ... ```) in expert response.")
                return {"error": "Invalid JSON structure", "raw_response": response_content}

        # Attempt to parse the extracted string
        parsed_json = json.loads(json_string)

        # Basic validation: Check if it's a dictionary
        if not isinstance(parsed_json, dict):
             logger.error(f"Parsed JSON is not a dictionary: {type(parsed_json)}")
             return {"error": "Parsed JSON is not a dictionary", "parsed_json_type": str(type(parsed_json)), "raw_response": response_content}

        # Optional: Further validation based on expected keys (could differ between modes)
        # Example check (adapt based on actual needs):
        # expected_keys_multi = ["问诊评分", "诊断依据评分", "诊断结果评分"]
        # expected_keys_one_step = ["诊断依据评分", "诊断结果评分"]
        # if not all(key in parsed_json for key in expected_keys_multi): # Adjust check based on mode if possible
        #      logger.warning(f"Parsed JSON missing expected keys: {parsed_json}")

        return parsed_json

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding expert evaluation JSON: {e}\nAttempted JSON string: '{json_string if json_string else 'N/A'}'")
        # Include raw response only if error is JSONDecodeError, otherwise it might be too large/redundant
        return {"error": "JSONDecodeError", "details": str(e), "raw_response": response_content if len(response_content)<2000 else response_content[:2000]+"..."}
    except Exception as e:
        # Catch other potential errors during parsing/regex
        logger.error(f"Unexpected error parsing expert evaluation: {e}")
        return {"error": "Unexpected parsing error", "details": str(e), "raw_response": response_content if len(response_content)<2000 else response_content[:2000]+"..."}


def extract_final_response(text: str) -> str:
    """
    Extracts the final response part if known markers like 'Final Response:' are present.
    Handles simple cases, might need refinement for complex model outputs.
    """
    markers = ["Final Response:", "Final Answer:"] # Add other markers if needed
    processed_text = text # Start with original text

    for marker in markers:
        # Case-insensitive find might be useful: lower_text = text.lower()
        idx = text.find(marker)
        if idx != -1:
            # Check if there's content after the marker
            content_start_idx = idx + len(marker)
            if content_start_idx < len(text):
                processed_text = text[content_start_idx:].strip()
                logger.debug(f"Extracted content after marker '{marker}'.")
                # If one marker is found, assume it's the final one and return
                return processed_text
            else:
                # Marker found but no content after it, potentially return empty or keep searching?
                logger.warning(f"Found marker '{marker}' but no content followed.")
                # Decide behavior: return "" or continue? Returning empty for now.
                return ""

    # Handle 'Thinking' specifically if needed, could be complex
    if "Thinking" in text and "Final Response" in text: # Re-check specific case from original code
        start_marker = "Final Response"
        start_idx = text.find(start_marker)
        if start_idx != -1 and start_idx + len(start_marker) < len(text):
            processed_text = text[start_idx + len(start_marker):].strip()
            logger.debug("Extracted content after 'Final Response' (with 'Thinking' present).")
            return processed_text
        else:
             logger.warning("Found 'Thinking' and 'Final Response' but failed extraction logic.")
             # Fallback to return original text minus the thinking part if possible? Difficult. Return original for now.
             return text


    # If no known markers were successfully processed, return the original text
    if processed_text is text:
         logger.debug("No known response markers found or processed.")

    return processed_text