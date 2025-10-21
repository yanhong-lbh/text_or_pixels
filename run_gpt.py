import os
import json
import argparse
import time
from PIL import Image as PIL_Image
import torch
import base64
import re
from openai import OpenAI


import json
from typing import List, Dict, Any

client = OpenAI()

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL (JSON Lines) file into a list of dictionaries.

    Each line in a JSONL file is expected to be a valid JSON object.

    Args:
        file_path: The full path to the .jsonl file.

    Returns:
        A list of dictionaries, where each dictionary represents a
        JSON object from a line in the file.
        
    Raises:
        FileNotFoundError: If the file at file_path does not exist.
        Exception: For other file reading errors.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # Skip empty lines
                if not line.strip():
                    continue
                try:
                    # Parse each line as a JSON object
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not decode JSON on line {line_number}: {e}")
                    print(f"         Line content: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise
        
    return data


def process_vision_info(messages):
    """
    Extracts image and video inputs from the message list.
    For this use case, it only handles images.
    """
    image_inputs = []
    video_inputs = [] 
    
    for message in messages:
        if message['role'] == 'user':
            content = message.get('content', [])
            for item in content:
                if item.get('type') == 'image':
                    img = item.get('image')
                    # If the 'image' is a file path, load it. Otherwise, assume it's a PIL Image.
                    if isinstance(img, str) and os.path.exists(img):
                        img = PIL_Image.open(img).convert("RGB")
                    elif isinstance(img, PIL_Image.Image):
                        pass
                    else:
                        print(f"Warning: Image item is not a valid path or PIL Image: {img}")
                        continue
                    image_inputs.append(img)
    return image_inputs, video_inputs


def call_gpt4_text(prompt: str, model: str = "gpt-4o-mini") -> (str, dict, float):
    """
    Send a text prompt to GPT-4 (text-only) and return the response.
    Returns a tuple: (output_text, usage_dict, latency_seconds).
    """
    start_time = time.time()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start_time
    output_text = response.output_text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    return output_text, usage, latency

def call_gpt4_vision(image_path: str, question: str, model: str = "gpt-4o-mini") -> (str, dict, float):
    """
    Send an image (by file path) plus a question prompt to GPT-4 with Vision.
    Returns (output_text, usage_dict, latency_seconds).
    """
    # Read image file and encode as base64 data URL
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    data_url = f"data:image/png;base64,{image_data}"

    # The OpenAI API expects the content to be an array of {"type": ..., ...} objects for multimodal inputs.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": question},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
                # {"type": "input_image", "image_url": data_url},
            ],
        }
    ]

    start_time = time.time()
    response = client.responses.create(model=model, input=messages)
    latency = time.time() - start_time
    output_text = response.output_text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    return output_text, usage, latency

def main():

    parser = argparse.ArgumentParser(description="Benchmark Gpt  on text vs. image inputs.")
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/ruler_niah_single_1_len500/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl",
        help="jsonl path"
        )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini", 
        help="The name of the gpt model to use."
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="images/ruler_niah_single_1/600_1000_500", 
        help="Directory containing the images of questions. Images should be named 'question_0.png', 'question_1.png', etc."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to process for each modality."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each response."
    )
    parser.add_argument(
    "--output", 
    type=str, 
    default="benchmark_results.json",
    help="json."
    )

    # Before running main, check if the image directory exists
    args = parser.parse_args()
    """
    Main function to run the benchmarking for text and image modalities.
    """

    print("--- Model and Processor Initialized ---")

    # --- Data Structures for Storing Results ---
    results = {
        "text_modality": [],
        "image_modality": [],
        "summary": {}
    }
    file_to_load = args.file_path

    loaded_data = load_jsonl(file_to_load)

    # ==============================================================================
    # 1. BENCHMARK TEXT INPUT RESPONSES
    # ==============================================================================
    print(f"\n--- Starting Text Modality Benchmark for {args.num_samples} Samples ---")
    text_total_time = 0
    text_total_input_tokens = 0
    total_correct_text = 0


    for i in range(args.num_samples):
        question_text = loaded_data[i]['doc']['input']
        question_text += ' Output the answer directly.'
        target = int(eval(loaded_data[i]['target'])[0])

        response_text, usage_text, latency_text = call_gpt4_text(prompt=question_text, model=args.model)

        numbers = re.findall(r'\d+', response_text)
        if numbers:
            extracted_answer = numbers[-1]
        else:
            extracted_answer = None
        try:
            extracted_answer = int(extracted_answer) if extracted_answer else None
        except:
            extracted_answer = None

        if extracted_answer == target:
            total_correct_text +=1

        
        
        input_token_count = usage_text["input_tokens"]
        text_total_input_tokens += input_token_count

        
        elapsed_time = latency_text
        text_total_time += latency_text

        # Record all data for this run
        results["text_modality"].append({
            "sample_index": i,
            "question": question_text,
            "response": response_text,
            "answer" : extracted_answer,
            "generation_time_sec": latency_text,
            "input_tokens": input_token_count,
            "target": target
        })
    
        print(f"Text Sample {i+1}/{args.num_samples} | Time: {elapsed_time:.2f}s | Input Tokens: {input_token_count}")

    # ==============================================================================
    # 2. BENCHMARK IMAGE INPUT RESPONSES
    # ==============================================================================
    print(f"\n--- Starting Image Modality Benchmark for {args.num_samples} Samples ---")
    image_total_time = 0
    image_total_input_tokens = 0
    total_correct = 0

    # This is the generic prompt used when the question is encoded in the image
    image_based_question = "Output the answer directly."

    for i in range(args.num_samples):
        target = int(eval(loaded_data[i]['target'])[0])
        image_path = os.path.join(args.image_dir, f"input_{i}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping sample {i}.")
            continue

        output_image, usage_image, latency_image=call_gpt4_vision(image_path=image_path,question=image_based_question,model=args.model)
        numbers = re.findall(r'\d+', output_image)
        if numbers:
            extracted_answer = numbers[-1]
        else:
            extracted_answer = None
        try:
            extracted_answer = int(extracted_answer) if extracted_answer else None
        except:
            extracted_answer = None


        input_token_count = usage_image['input_tokens']
        image_total_input_tokens += input_token_count
        image_total_time += latency_image
        if extracted_answer == target:
            total_correct +=1





        # Record all data for this run
        results["image_modality"].append({
            "sample_index": i,
            "image_path": image_path,
            "response": output_image,
            'answer': extracted_answer,
            "generation_time_sec": latency_image,
            "input_tokens": input_token_count,
            "target": target
        })
        print(f"Image Sample {i+1}/{args.num_samples} | Time: {latency_image:.2f}s | Input Tokens: {input_token_count}")

    # ==============================================================================
    # 3. CALCULATE AVERAGES AND REPORT
    # ==============================================================================
    print("\n--- Benchmark Complete. Generating Summary ---")
    
    # Calculate averages, avoiding division by zero
    avg_text_time = text_total_time / len(results["text_modality"]) if results["text_modality"] else 0
    avg_text_tokens = text_total_input_tokens / len(results["text_modality"]) if results["text_modality"] else 0
    avg_image_time = image_total_time / len(results["image_modality"]) if results["image_modality"] else 0
    avg_image_tokens = image_total_input_tokens / len(results["image_modality"]) if results["image_modality"] else 0

    results["summary"] = {
        "text_modality_summary": {
            "total_samples": len(results["text_modality"]),
            "average_generation_time_sec": avg_text_time,
            "average_input_tokens": avg_text_tokens,
            "accuracy": total_correct_text / len(results["text_modality"]),
            "correct_count": total_correct_text,
            
        },
        "image_modality_summary": {
            "total_samples": len(results["image_modality"]),
            "average_generation_time_sec": avg_image_time,
            "average_input_tokens": avg_image_tokens,
            "accuracy": total_correct / len(results["image_modality"]),
            "correct_count": total_correct,

        }
    }

    # Print summary to console
    print("\n================= BENCHMARK SUMMARY =================")
    print("\n[Text Modality]")
    print(f"  Total Samples Processed: {results['summary']['text_modality_summary']['total_samples']}")
    print(f"  Accuracy: {total_correct_text}/{len(results['text_modality'])} = {total_correct_text/len(results['text_modality']):.2%}")
    print(f"  Average Generation Time: {avg_text_time:.4f} seconds")
    print(f"  Average Input Tokens:    {avg_text_tokens:.2f} tokens")
    
    print("\n[Image Modality]")
    print(f"  Total Samples Processed: {results['summary']['image_modality_summary']['total_samples']}")
    print(f"  Accuracy: {total_correct}/{len(results['image_modality'])} = {total_correct/len(results['image_modality']):.2%}")
    print(f"  Average Generation Time: {avg_image_time:.4f} seconds")
    print(f"  Average Input Tokens:    {avg_image_tokens:.2f} tokens")
    print("\n===================================================")

    # Save detailed results to a JSON file
    output_filename = args.output
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll detailed results saved to '{output_filename}'")


if __name__ == "__main__":
    main()