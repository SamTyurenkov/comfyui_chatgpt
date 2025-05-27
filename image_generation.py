import base64
import os
from io import BytesIO
from PIL import Image
import numpy
from openai import OpenAI
import torch
from dotenv import load_dotenv

load_dotenv()

# https://pypi.org/project/openai/

class ImageToBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "optional": {
            "image": ("IMAGE",),
        },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_base64"
    OUTPUT_NODE = True

    def to_base64(self, image, ):
      import base64
      from io import BytesIO
      
      def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))
      
      pil_image = tensor2pil(image)
      buffered = BytesIO()
      pil_image.save(buffered, format="PNG")
      image_bytes = buffered.getvalue()

      base64_str = base64.b64encode(image_bytes).decode("utf-8")
      return {"result": (base64_str,)}

class ChatGPTImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        SIZE_MODES = ["auto","1024x1024", "1024x1536", "1536x1024"]
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image based on provided image(s).", "multiline": True}),
                "size": (SIZE_MODES, {"default":"auto"}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "response_id": ("STRING",)
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, size, moderation, quality, image1=None, image2=None, response_id=None):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = Image.open(BytesIO(image_bytes))
            return image_data
        
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        messages = []
        if not response_id:
            messages.append({"role": "system", "content": "You assist in image generations and edits"})

        messages.append(
            {
                "role": "user",
                "content": []
            })

        if image1 and not response_id:
            messages[1]["content"].append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image1}"
            })

        if image2 and not response_id:
            messages[1]["content"].append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image2}"
            })

        if response_id:
            messages[0]["content"].append({"type": "input_text", "text": prompt})
        else:
            messages[1]["content"].append({"type": "input_text", "text": prompt})

        request_args = {
            "model":"gpt-4.1-mini",
            "input":messages,
            "tools":[
                {
                    "type": "image_generation",
                    "size": size,
                    "moderation": moderation,
                    "quality": quality
                 }
            ]
        }

        if response_id:
            request_args["previous_response_id"] = response_id

        response = client.responses.create(
            **request_args
        )

        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]
        
        if image_data:
            image_base64 = image_data[0]
            image_data = base64_to_image(image_base64)
            output_t = pil2tensor(image_data)
        else:
            output_t = empty_image()

        result = str(response)
        return (output_t, result, response.id)