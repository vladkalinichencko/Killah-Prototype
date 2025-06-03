\
# filepath: /Users/vladislavkalinichenko/XcodeProjects/Killah Prototype/Resources/text_transformer_llm.py
import sys
import time
import json
import random

def simulate_llm_transformation(text_to_transform, user_prompt):
    """
    Simulates an LLM transforming text based on a prompt.
    In a real application, this script would:
    1. Load an LLM.
    2. Construct a detailed prompt including `text_to_transform` and `user_prompt`.
    3. Generate the transformed text.
    4. Print the transformed text to stdout.
    """
    print(f"Python LLM Transformer: Received text (first 50 chars): '{text_to_transform[:50]}...'", flush=True)
    print(f"Python LLM Transformer: Received user prompt: '{user_prompt}'", flush=True)

    # Simulate processing time
    time.sleep(random.uniform(0.5, 1.5))

    # Simulate transformation
    # This is highly simplified. A real LLM would do complex changes.
    transformed_text = text_to_transform # Start with original
    
    if "fix" in user_prompt.lower() or "change" in user_prompt.lower():
        if "дед да бабка" in text_to_transform and "брат с сестрой" in user_prompt:
            transformed_text = text_to_transform.replace("дед да бабка", "брат с сестрой [LLM CHANGE]")
        else:
            transformed_text = f"[LLM APPLIED CHANGE BASED ON '{user_prompt}'] " + text_to_transform
    elif "дополни" in user_prompt.lower() or "продолжаем" in user_prompt.lower():
        transformed_text = text_to_transform + f" [LLM ADDITION BASED ON '{user_prompt}'] ... and so on."
    else:
        transformed_text = text_to_transform + " [LLM PROCESSED]"
        
    # To demonstrate a more structured "diff" potential,
    # we could output original and transformed if the Swift side expects it.
    # For now, just the transformed text.
    return transformed_text

if __name__ == "__main__":
    print("Python LLM Transformer: Ready for text and prompt.", flush=True)
    try:
        while True:
            # Expecting a JSON object with "text" and "prompt"
            line = sys.stdin.readline()
            if not line:
                print("Python LLM Transformer: stdin closed, exiting.", flush=True)
                break
            
            try:
                input_data = json.loads(line)
                text = input_data.get("text")
                prompt = input_data.get("prompt", "Process this text.") # Default prompt
                
                if text is None:
                    print("Python LLM Transformer: Error - 'text' not found in input JSON. Skipping.", flush=True)
                    continue

                result = simulate_llm_transformation(text, prompt)
                # Output as JSON for easier parsing on Swift side, including original for diffing
                output_json = json.dumps({"original": text, "transformed": result})
                print(output_json, flush=True)

            except json.JSONDecodeError:
                print(f"Python LLM Transformer: Error decoding JSON from input line: {line.strip()}", flush=True)
            except Exception as e:
                print(f"Python LLM Transformer: Error during processing: {e}", flush=True)

    except KeyboardInterrupt:
        print("Python LLM Transformer: Interrupted. Exiting.", flush=True)
    finally:
        print("Python LLM Transformer: Shutting down.", flush=True)
