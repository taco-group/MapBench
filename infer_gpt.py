import argparse
import torchvision.transforms as T
import time
import warnings
warnings.filterwarnings("ignore")
import random
import json
import requests as requests
import base64
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import re
import os
random.seed(42)


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str


def gpt4o_mini(question, pil_img=None, proxy='openai', args=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + args.api_key,
    }

    if pil_img is not None:
        base64_image = pil_to_base64(pil_img)
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "model": args.gpt_model if args.gpt_model else 'gpt-4o-mini-2024-07-18'
        }
    else:
        params = {
            "messages": [

                {
                    "role": 'user',
                    "content": question
                }
            ],
            "model": args.gpt_model if args.gpt_model else 'gpt-4o-mini-2024-07-18'
        }
    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content





def extract_navigation_steps(full_instructions):
    simplified_steps = []
    lines = full_instructions.split('\n')
    for line in lines:
        if len(re.findall(r'->', line)) == 1:
            line = re.sub(r'^[\W_]+|[\W_]+$', '', line)
            if "next node" in line.lower():
                return []
            simplified_steps.append(line.replace("*", ""))
    return simplified_steps


def append_to_json(save_dir, new_entry):
    if os.path.exists(save_dir):
        with open(save_dir, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'runs' not in data:
                    data = {'runs': []}
            except json.JSONDecodeError:
                data = {'runs': []}
    else:
        data = {'runs': []}

    data['runs'].append(new_entry)

    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    dataset = load_dataset('shuoxing/MapBench_VQA', split='test')

    for map_data in dataset:

        pil_img = map_data["image"]
        queries = map_data["queries"]
        map_class = map_data["map_class"]
        save_dir = os.path.join(args.save_dir, f"{map_class}.jsonl")

        for query in queries:
            start = query["start"]
            destination = query["destination"]
            gpt_answer = query["gpt_answer"]
            
            prompt = f"""
                    Please provide the simplest direct path from {start} to {destination}, focusing only on visible landmarks and roads.
                    Provide the instructions in the format: "Start Node Name -> Next Node Name (from Direction, moving along Road Name/Landmark Name) \n ... \nNode Name -> Destination Node Name (from Direction, moving along Road Name/Landmark Name) \n", and ensure the instructions are clear and direct.
                    """

            for i in range(5):
                response = gpt4o_mini(prompt, pil_img=pil_img, args=args)
                simplified_steps = extract_navigation_steps(response)
                if len(simplified_steps) > 0:
                    break

            if len(simplified_steps) == 0:
                simplified_steps = ["No path found"]

            new_entry = {
                "image_id": map_data["image_id"],
                "map_class": map_class,
                "start": start,
                "destination": destination,
                "gpt_answer": gpt_answer,
                "answer": simplified_steps
            }

            with open(save_dir,"a") as f:
                f.write(json.dumps(new_entry))
                f.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, default=None)
    args = parser.parse_args()

    main(args)