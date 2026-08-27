import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ChatGPTTextGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "gpt-5.4",
            "gpt-5.2",
            "gpt-5",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5-nano",
            "o3",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        image_detail_modes = ["auto", "low", "high"]
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "Describe the provided image(s).", "multiline": True},
                ),
                "model": (models, {"default": "gpt-4.1-mini"}),
            },
            "optional": {
                "instructions": ("STRING", {"default": "", "multiline": True}),
                "image1": ("STRING",),
                "image2": ("STRING",),
                "image3": ("STRING",),
                "image4": ("STRING",),
                "image_detail": (image_detail_modes, {"default": "auto"}),
                "response_id": ("STRING",),
                "max_output_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 128000},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "response", "response_id")
    FUNCTION = "request"

    def request(
        self,
        prompt,
        model,
        instructions="",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image_detail="auto",
        response_id=None,
        max_output_tokens=4096,
    ):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        content = [{"type": "input_text", "text": prompt}]
        for image in (image1, image2, image3, image4):
            if not image:
                continue
            image = image.strip()
            image_url = (
                image if image.startswith("data:") else f"data:image/png;base64,{image}"
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": image_detail,
                }
            )

        request_args = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        if instructions:
            request_args["instructions"] = instructions
        if response_id:
            request_args["previous_response_id"] = response_id

        response = client.responses.create(**request_args)
        return (response.output_text, str(response), response.id)
