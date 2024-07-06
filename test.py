import requests
import json
from typing import List
from PIL import Image
import io
import base64

BASE_URL = ""  # Replace with your actual FastAPI server URL

def send_chat_requests(
        prompts: List[dict],
        model_selector: str = "idefics2-8b-chatty",
        decoding_strategy: str = "Top P Sampling",
        temperature: float = 0.7,
        max_new_tokens: int = 100,
        repetition_penalty: float = 1.0,
        top_p: float = 0.9
) -> List[str]:
    """
    Send multiple chat requests to the FastAPI IDEFICS2 Chat Application.
    """
    url = f"{BASE_URL}/chat"
    payload = {
        "prompts": prompts,
        "model_selector": model_selector,
        "decoding_strategy": decoding_strategy,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["responses"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def image_to_base64(image_path):
    """
    Convert an image file to a base64-encoded string.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


if __name__ == "__main__":

    # Convert image to base64
    # add the image path here
    image_base64 = image_to_base64("")

    prompts = [
        {
            "text": "what do u see in the image ?",
            "files": [image_base64]  # Send base64-encoded image
        },
        {
            "text": "i want u to call him as mclane",
            "files": []
        },
        {
            "text": "what colour is his jersey?",
            "files": []
        },
        {
            "text": "tell me his name?",
            "files": []
        }
    ]

    try:
        # Send multiple prompts
        responses = send_chat_requests(prompts)
        print("Model Responses:")
        for i, response in enumerate(responses, 1):
            print(f"\nResponse {i}:")
            print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
