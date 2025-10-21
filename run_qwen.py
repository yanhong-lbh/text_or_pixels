import os
import json
import argparse
import time
from PIL import Image as PIL_Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import json
from typing import List, Dict, Any

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
                    if isinstance(img, str) and os.path.exists(img):
                        img = PIL_Image.open(img).convert("RGB")
                    elif isinstance(img, PIL_Image.Image):
                        pass
                    else:
                        print(f"Warning: Image item is not a valid path or PIL Image: {img}")
                        continue
                    image_inputs.append(img)
    return image_inputs, video_inputs

def main(args):
    """
    Main function to run the benchmarking for text and image modalities.
    """
    print("--- Initializing Model and Processor ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto", 
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    print("--- Model and Processor Initialized ---")

    # --- Data Structures for Storing Results ---
    results = {
        "text_modality": [],
        "image_modality": [],
        "summary": {}
    }
    # file_to_load = '/share/data/speech/yanhong/zx/1000_data/samples_niah_single_1_2025-06-27T14-58-32.284616.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/1500_data/samples_niah_single_1_2025-06-27T15-34-58.043188.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/2500_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-06-28T03-47-53.308664.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/2000_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-06-28T03-45-21.943532.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/3000_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-06-28T04-08-03.348492.jsonl'
    file_to_load = '/share/data/speech/yanhong/zx/3500_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-06-28T04-57-24.048812.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/4000_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-06-29T01-46-42.271862.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/1212_data/Qwen__Qwen2.5-0.5B-Instruct//samples_niah_single_1_2025-07-09T15-27-45.000598.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/1195_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-07-09T17-42-54.411946.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/981_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-07-09T17-39-56.462136.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/836_data/Qwen__Qwen2.5-0.5B-Instruct/samples_niah_single_1_2025-07-09T17-38-07.788292.jsonl'
    # file_to_load = '/share/data/speech/yanhong/zx/500_data/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl'

    loaded_data = load_jsonl(file_to_load)

    # ==============================================================================
    # 1. BENCHMARK TEXT INPUT RESPONSES
    # ==============================================================================
    print(f"\n--- Starting Text Modality Benchmark for {args.num_samples} Samples ---")
    text_total_time = 0
    text_total_input_tokens = 0

    for i in range(args.num_samples):
        question_text = loaded_data[i]['doc']['input']
        question_text += ' Output the answer directly.'
        target = int(eval(loaded_data[i]['target'])[0])

        # Construct the message payload for a text-only query
        messages_text = [
            {"role": "user", "content": [{"type": "text", "text": question_text}]}
        ]

        # Prepare input for the model
        text_prompt = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
        # For text-only, image_inputs will be an empty list
        image_inputs, video_inputs = process_vision_info(messages_text)

        inputs = processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        input_token_count = inputs.input_ids.shape[-1]
        text_total_input_tokens += input_token_count

        # Start timer and generate response
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        text_total_time += elapsed_time

        # Decode the generated output
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        acc = 1 if str(output_text).strip() == str(target) else 0
        # Record all data for this run
        results["text_modality"].append({
            "sample_index": i,
            "question": question_text,
            "response": output_text,
            "generation_time_sec": elapsed_time,
            "input_tokens": input_token_count,
            "target": target,
            "acc": acc
        })
        print(f"Text Sample {i+1}/{args.num_samples} | Time: {elapsed_time:.2f}s | Input Tokens: {input_token_count}")

    # ==============================================================================
    # 2. BENCHMARK IMAGE INPUT RESPONSES
    # ==============================================================================
    print(f"\n--- Starting Image Modality Benchmark for {args.num_samples} Samples ---")
    image_total_time = 0
    image_total_input_tokens = 0

    # This is the generic prompt used when the question is encoded in the image
    image_based_question = "Output the answer directly."

    for i in range(args.num_samples):
        target = int(eval(loaded_data[i]['target'])[0])
        image_path = os.path.join(args.image_dir, f"input_{i}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping sample {i}.")
            continue

        # Load the image
        try:
            img = PIL_Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping sample {i}.")
            continue

        # Construct the message payload for an image query
        messages_image = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": image_based_question}
                ],
            }
        ]
        
        # Prepare input for the model
        text_prompt_img = processor.apply_chat_template(messages_image, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = processor(
            text=[text_prompt_img],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        input_token_count = inputs.input_ids.shape[-1]
        image_total_input_tokens += input_token_count

        # Start timer and generate response
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        image_total_time += elapsed_time

        # Decode the generated output
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        acc = 1 if str(output_text).strip() == str(target) else 0
        # Record all data for this run
        results["image_modality"].append({
            "sample_index": i,
            "image_path": image_path,
            "response": output_text,
            "generation_time_sec": elapsed_time,
            "input_tokens": input_token_count,
            "target": target,
            "acc": acc
        })
        print(f"Image Sample {i+1}/{args.num_samples} | Time: {elapsed_time:.2f}s | Input Tokens: {input_token_count}")

    # ==============================================================================
    # 3. CALCULATE AVERAGES AND REPORT
    # ==============================================================================
    print("\n--- Benchmark Complete. Generating Summary ---")
    
    # Calculate averages, avoiding division by zero
    avg_text_time = text_total_time / len(results["text_modality"]) if results["text_modality"] else 0
    avg_text_tokens = text_total_input_tokens / len(results["text_modality"]) if results["text_modality"] else 0
    avg_image_time = image_total_time / len(results["image_modality"]) if results["image_modality"] else 0
    avg_image_tokens = image_total_input_tokens / len(results["image_modality"]) if results["image_modality"] else 0
    # === ADDED: Compute mean accuracy for each modality ===
    avg_text_acc = sum([x['acc'] for x in results["text_modality"]]) / len(results["text_modality"]) if results["text_modality"] else 0
    avg_image_acc = sum([x['acc'] for x in results["image_modality"]]) / len(results["image_modality"]) if results["image_modality"] else 0


    results["summary"] = {
        "text_modality_summary": {
            "total_samples": len(results["text_modality"]),
            "average_generation_time_sec": avg_text_time,
            "average_input_tokens": avg_text_tokens,
            "average_accuracy": avg_text_acc,
        },
        "image_modality_summary": {
            "total_samples": len(results["image_modality"]),
            "average_generation_time_sec": avg_image_time,
            "average_input_tokens": avg_image_tokens,
            "average_accuracy": avg_image_acc,
        }
    }

    # Print summary to console
    print("\n================= BENCHMARK SUMMARY =================")
    print("\n[Text Modality]")
    print(f"  Total Samples Processed: {results['summary']['text_modality_summary']['total_samples']}")
    print(f"  Average Generation Time: {avg_text_time:.4f} seconds")
    print(f"  Average Input Tokens:    {avg_text_tokens:.2f} tokens")
    
    print("\n[Image Modality]")
    print(f"  Total Samples Processed: {results['summary']['image_modality_summary']['total_samples']}")
    print(f"  Average Generation Time: {avg_image_time:.4f} seconds")
    print(f"  Average Input Tokens:    {avg_image_tokens:.2f} tokens")
    print("\n===================================================")

    # Save detailed results to a JSON file
    output_filename = "benchmark_results_750_1250_3500.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll detailed results saved to '{output_filename}'")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Benchmark Qwen-VL model on text vs. image inputs.")
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/ruler_niah_single_1_len500/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl",
        help="jsonl path"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-VL-72B-Instruct", 
        help="The name of the Hugging Face model to use."
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

    parsed_args = parser.parse_args()
    if not os.path.isdir(parsed_args.image_dir):
        print(f"Error: Image directory '{parsed_args.image_dir}' not found.")
        print("Please create it and place your pre-generated question images inside.")
        os.makedirs(parsed_args.image_dir, exist_ok=True)
        print(f"Created directory '{parsed_args.image_dir}'. Make sure to populate it with images.")
    
    main(parsed_args)