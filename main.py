from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Union
import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from PIL import Image
import io
import urllib.request
import copy
import base64

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User's questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?",
            },
        ],
    }
]

# Global variable to store chat history
CHAT_HISTORY = []


class UserPrompt(BaseModel):
    text: str
    files: List[str] = []  # This can now contain URLs or base64-encoded images


class InferenceParams(BaseModel):
    prompts: List[UserPrompt]
    model_selector: str
    decoding_strategy: str
    temperature: float
    max_new_tokens: int
    repetition_penalty: float
    top_p: float


def load_image(image_data):
    if image_data.startswith('http://') or image_data.startswith('https://'):
        # It's a URL
        with urllib.request.urlopen(image_data) as response:
            image_data = response.read()
    else:
        # Assume it's base64 encoded
        try:
            image_data = base64.b64decode(image_data)
        except:
            raise HTTPException(status_code=400, detail="Invalid image data. Must be a URL or base64 encoded image.")

    image_stream = io.BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def format_user_prompt_with_im_history_and_system_conditioning(
        user_prompt: UserPrompt, chat_history: List[List[Union[str, None]]]
) -> tuple[List[Dict[str, Union[List, str]]], List[Image.Image]]:
    resulting_messages = copy.deepcopy(SYSTEM_PROMPT)
    resulting_images = []

    # Format history
    for turn in chat_history:
        if not resulting_messages or resulting_messages[-1]["role"] != "user":
            resulting_messages.append({"role": "user", "content": []})

        if turn[1] is None:  # Pure media turn
            resulting_messages[-1]["content"].append({"type": "image"})
            resulting_images.append(load_image(turn[0]))
        else:
            user_utterance, assistant_utterance = turn
            resulting_messages[-1]["content"].append({"type": "text", "text": user_utterance.strip()})
            resulting_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_utterance.strip()}],
            })

    # Format current input
    if not user_prompt.files:
        resulting_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompt.text}],
        })
    else:
        resulting_messages.append({
            "role": "user",
            "content": [{"type": "image"}] * len(user_prompt.files) + [{"type": "text", "text": user_prompt.text}],
        })
        resulting_images.extend([load_image(path) for path in user_prompt.files])

    return resulting_messages, resulting_images


def model_inference(params: InferenceParams):
    global CHAT_HISTORY
    CHAT_HISTORY = []

    all_responses = []

    for user_prompt in params.prompts:
        if user_prompt.text.strip() == "" and not user_prompt.files:
            raise HTTPException(status_code=400, detail="Please input a query and optionally image(s).")

        if user_prompt.text.strip() == "" and user_prompt.files:
            raise HTTPException(status_code=400, detail="Please input a text query along the image(s).")

        generation_args = {
            "max_new_tokens": params.max_new_tokens,
            "repetition_penalty": params.repetition_penalty,
        }

        if params.decoding_strategy == "Greedy":
            generation_args["do_sample"] = False
        elif params.decoding_strategy == "Top P Sampling":
            generation_args["temperature"] = params.temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = params.top_p
        else:
            raise HTTPException(status_code=400, detail="Invalid decoding strategy")

        resulting_text, resulting_images = format_user_prompt_with_im_history_and_system_conditioning(
            user_prompt=user_prompt,
            chat_history=CHAT_HISTORY,
        )
        prompt = PROCESSOR.apply_chat_template(resulting_text, add_generation_prompt=True)
        inputs = PROCESSOR(
            text=prompt,
            images=resulting_images if resulting_images else None,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        generation_args.update(inputs)

        generated_ids = MODELS[params.model_selector].generate(**generation_args)
        generated_text = \
        PROCESSOR.batch_decode(generated_ids[:, inputs["input_ids"].size(-1):], skip_special_tokens=True)[0]

        # Update chat history
        CHAT_HISTORY.append([user_prompt.text, generated_text.strip()])
        all_responses.append(generated_text.strip())

    return all_responses


@app.post("/chat")
async def chat(params: InferenceParams):
    responses = model_inference(params)
    return JSONResponse(content={"responses": responses})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
