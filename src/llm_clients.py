# src/llm_clients.py
import time
import logging
from openai import OpenAI, RateLimitError, APIError, OpenAIError, APIConnectionError, APITimeoutError
from typing import List, Dict, Any, Optional
# Import utility for post-processing if needed here
from .utils import extract_final_response

# Use logging configured in main.py
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1 # Start with 1 second for the first retry sleep calculation
BACKOFF_FACTOR = 3 # Increase sleep time (e.g., 1, 3, 9, 27, 81 secs)

# --- Client Initialization ---
def initialize_openai_client(base_url: Optional[str], api_key: Optional[str]) -> Optional[OpenAI]:
    """Initializes and returns an OpenAI client."""
    if not base_url:
        logger.error("Missing base_url for OpenAI client.")
        return None
    # Allow initialization without key for local models that don't require one
    if not api_key:
        logger.warning(f"API key is missing or empty for base_url {base_url}. Client initialized without key (may fail if key is required).")

    try:
        # Consider adding timeout configuration
        # Example: client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0) # Handle retries manually below
        client = OpenAI(base_url=base_url, api_key=api_key)
        # Optional: Add a health check ping here if the API supports it and it's cheap
        # try:
        #     client.models.list() # Example check - might incur cost/rate limit
        #     logger.info(f"Successfully listed models from base_url: {base_url}")
        # except Exception as ping_error:
        #      logger.warning(f"Could not verify connection to {base_url} (model list failed): {ping_error}")
        logger.info(f"OpenAI client initialized for base_url: {base_url}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client for base_url {base_url}: {e}")
        return None


# --- Completion Creation with Retries ---
def create_llm_completion(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: Optional[int] = 1024,
    stop: Optional[List[str]] = None,
    stream: bool = False,
    max_retries: int = MAX_RETRIES
) -> Optional[Any]:
    """
    Creates a chat completion using the OpenAI client with retry logic
    for specific retryable errors.

    Returns:
        The completion object (non-stream) or stream iterator, or None if fails after retries.
    """
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            # Log message count and maybe first/last role for debugging context length issues
            log_msg_summary = f"Messages: {len(messages)}"
            if messages: log_msg_summary += f" (First: {messages[0]['role']}, Last: {messages[-1]['role']})"
            logger.debug(f"Attempting API call to {model_name} (Attempt {retries + 1}/{max_retries}). {log_msg_summary}")

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop if stop else None, # Pass None if stop list is empty/None
                stream=stream
                # Add other parameters like top_p if needed
            )
            # If call succeeds, return completion immediately
            return completion
        except RateLimitError as e:
            last_exception = e
            wait_time = (INITIAL_BACKOFF * (BACKOFF_FACTOR ** retries))
            logger.warning(f"Rate limit error for {model_name}: {e}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
        except (APIConnectionError, APITimeoutError) as e:
             # Retry on potentially transient network errors
            last_exception = e
            wait_time = (INITIAL_BACKOFF * (BACKOFF_FACTOR ** retries))
            logger.warning(f"API connection/timeout error for {model_name}: {e}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
        except OpenAIError as e:
             # Catch other potentially retryable OpenAI errors (e.g., temporary service issues)
             # Check e.status_code if needed (e.g., 5xx errors are often retryable)
             if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                 last_exception = e
                 wait_time = (INITIAL_BACKOFF * (BACKOFF_FACTOR ** retries))
                 logger.warning(f"OpenAI Service Error ({e.status_code}) for {model_name}: {e}. Retrying in {wait_time:.2f} seconds...")
                 time.sleep(wait_time)
                 retries += 1
             else: # Treat other OpenAIErrors as non-retryable for now
                 logger.error(f"OpenAI Error (non-retryable or status unknown) for {model_name}: {e}")
                 last_exception = e
                 break # Exit retry loop
        except APIError as e:
            # Non-retryable API errors (e.g., authentication, bad request 4xx)
            logger.error(f"API Error (non-retryable) for {model_name}: {e}. Check request details/API key.")
            last_exception = e
            break # Exit retry loop
        except Exception as e:
            # Catch any other unexpected errors during the API call itself
            logger.exception(f"An unexpected error occurred during API call to {model_name}: {e}") # Use logger.exception to include traceback
            last_exception = e
            break # Exit retry loop

    logger.error(f"API call failed for model {model_name} after {max_retries} retries. Last exception: {last_exception}")
    return None


# --- Stream Handling ---
def handle_stream_response(completion_stream) -> str:
    """Collects content from a streaming completion."""
    full_content = ""
    try:
        chunk_count = 0
        for chunk in completion_stream:
            chunk_count += 1
            # Check structure carefully based on streaming API version
            if hasattr(chunk, 'choices') and chunk.choices and \
               hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta and \
               hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            # Add handling for finish_reason if needed
        logger.debug(f"Collected stream response from {chunk_count} chunks.")
    except Exception as e:
        logger.error(f"Error processing stream chunk: {e}")
        # Return partially collected content or empty string? Return partial for now.
    return full_content

# --- Generic Function to Get LLM Response String ---
def get_llm_response(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    config, # Pass config for model-specific settings
    is_test_model: bool = False # Indicate if it's the model under test
    ) -> Optional[str]:
    """
    Gets a response string from the LLM, handling retries, streaming, fallbacks, and post-processing.
    """
    # Determine parameters based on model type and config
    # Note: Using output_json_path in config for special model check - refine if possible
    output_path_str = getattr(config, 'output_json_path', '') # Safely get attribute
    is_special_model = ("lingdan" in output_path_str or \
                        "HuatuoGPT" in output_path_str) and is_test_model

    max_tokens = 256 if is_special_model else 1024
    # Safely get test_model_name
    test_model_name_str = getattr(config, 'test_model_name', '')
    stop_sequences = ["<|im_end|>"] if is_special_model else None
    use_stream = "qwen3" in test_model_name_str and is_test_model

    completion = None
    # --- Primary API Call ---
    try:
        completion = create_llm_completion(
            client=client,
            model_name=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
            stop=stop_sequences,
            stream=use_stream
        )
    except Exception as e:
         # Catch exceptions *during* the create_llm_completion call itself if they bypass internal handling
         logger.error(f"Direct exception during primary API call wrapper for {model_name}: {e}")
         completion = None

    # --- Fallback Logic for Test Model (if applicable and needed) ---
    if completion is None and is_test_model and len(messages) > 1:
        logger.warning(f"Primary API call failed for test model {model_name}. Retrying with shortened history.")
        # Construct fallback message list: system prompt(s) + last user message
        short_messages = [msg for msg in messages if msg['role'] == 'system'] # Keep all system prompts
        last_user_msg = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)

        if last_user_msg:
            short_messages.append(last_user_msg)
            if len(short_messages) > (len(messages) - len([m for m in messages if m['role']!='system' and m['role']!='user'])): # Basic check if history is shorter
                try:
                    completion = create_llm_completion( # Call API again
                        client=client, model_name=model_name, messages=short_messages,
                        temperature=0.3, max_tokens=max_tokens, stop=stop_sequences, stream=use_stream
                    )
                except Exception as fb_e:
                     logger.error(f"Direct exception during fallback API call wrapper for {model_name}: {fb_e}")
                     completion = None # Ensure completion is None if exception occurs here
            else: logger.error("Could not construct a valid shortened history for fallback.")
        else: logger.error("No user message found for fallback history construction.")


    # --- Process Completion (if any attempt was successful) ---
    if completion is None:
        logger.error(f"API response is None for model {model_name} after primary attempt (and fallback if applicable).")
        return None # Explicitly return None if completion object is None

    response_content = None
    try:
        if use_stream:
            response_content = handle_stream_response(completion)
        else:
            # Handle potential variations in non-streaming response structure more carefully
            if hasattr(completion, 'choices') and completion.choices:
                # Check if message attribute exists and has content
                 if hasattr(completion.choices[0], 'message') and completion.choices[0].message and \
                    hasattr(completion.choices[0].message, 'content'):
                     response_content = completion.choices[0].message.content
                 # Add check for function call or other response types if needed
                 # elif hasattr(completion.choices[0], 'finish_reason') and completion.choices[0].finish_reason == 'function_call':
                 #     logger.warning("Model returned a function call, not content.") # Handle appropriately
                 #     return None # Or return function call info
                 else: # Choice exists but no usable message content found
                     logger.warning(f"Completion choice for {model_name} lacks valid message content: {completion.choices[0]}")
            else:
                 logger.error(f"Invalid non-streaming response structure (no choices) for model {model_name}: {completion}")
                 return None

        # Check if content extraction was successful
        if response_content is None:
             # This might happen if the message content was None/empty or structure was unexpected
             logger.error(f"Response content could not be extracted after processing completion for {model_name}.")
             return None

        # Post-process known model-specific markers/outputs
        processed_content = extract_final_response(response_content) # Assumes extract_final_response is available (from utils)
        return processed_content.strip()

    except Exception as e:
         # Catch errors during stream handling or content extraction
         logger.exception(f"Error processing completion response for model {model_name}: {e}") # Use exception to get traceback
         return None