import base64
import os
from io import BytesIO
from PIL import Image, ImageOps
import numpy
from openai import OpenAI
import torch
from dotenv import load_dotenv
from .utils import ImageToBase64

load_dotenv()

# https://pypi.org/project/openai/

class ChatGPTImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        SIZE_MODES = ["auto","1024x1024", "1024x1536", "1536x1024"]
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high"]
        IMAGE_TOOL_MODELS = [
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
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image based on provided image(s).", "multiline": True}),
                "model": (IMAGE_TOOL_MODELS, {"default": "gpt-4.1-mini"}),
                "size": (SIZE_MODES, {"default":"auto"}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "response_id": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, moderation, quality, image1=None, image2=None, response_id=None, seed=0):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
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
            "model": model,
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

        image_results = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]
        
        if image_results:
            output_t = torch.cat(
                [pil2tensor(base64_to_image(image_base64)) for image_base64 in image_results],
                dim=0,
            )
        else:
            output_t = empty_image()

        result = str(response)
        return (output_t, result, response.id)


class ChatGPTImageModelGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        SIZE_MODES = ["auto","1024x1024", "1024x1536", "1536x1024"]
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high"]
        IMAGE_MODELS = ["gpt-image-1", "gpt-image-1.5", "gpt-image-2"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image according to this prompt.", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2"}),
                "size": (SIZE_MODES, {"default":"auto"}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, moderation, quality, n, seed=0):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
            return image_data
        
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        request_args = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "moderation": moderation,
            "quality": quality,
            "n": n,
        }
        if seed:
            request_args["seed"] = seed

        response = client.images.generate(
            **request_args
        )

        image_results = [
            item.b64_json
            for item in response.data
            if getattr(item, "b64_json", None)
        ]

        if image_results:
            output_t = torch.cat(
                [pil2tensor(base64_to_image(image_base64)) for image_base64 in image_results],
                dim=0,
            )
        else:
            output_t = empty_image()

        result = str(response)
        return (output_t, result)


class ChatGPTImageEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        SIZE_MODES = ["auto","1024x1024", "1024x1536", "1536x1024"]
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high"]
        INPUT_FIDELITY_MODES = ["high", "low"]
        IMAGE_MODELS = ["gpt-image-1", "gpt-image-1.5", "gpt-image-2"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit image according to this prompt.", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2"}),
                "size": (SIZE_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"}),
                "input_fidelity": (INPUT_FIDELITY_MODES, {"default":"low"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "mask": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, quality, input_fidelity, n, image1=None, image2=None, mask=None, seed=0):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
            return image_data

        def base64_to_file(b64_string):
            file = BytesIO(base64.b64decode(b64_string))
            file.name = "image.png"
            return file

        def mask_b64_to_file(b64_string):
            raw = Image.open(BytesIO(base64.b64decode(b64_string)))
            if raw.mode in ("RGBA", "LA"):
                mask_rgba = raw.convert("RGBA")
            else:
                mask_l = raw.convert("L")
                alpha = ImageOps.invert(mask_l)
                mask_rgba = Image.merge(
                    "RGBA",
                    (
                        Image.new("L", mask_l.size, 255),
                        Image.new("L", mask_l.size, 255),
                        Image.new("L", mask_l.size, 255),
                        alpha,
                    ),
                )
            buf = BytesIO()
            mask_rgba.save(buf, format="PNG")
            buf.seek(0)
            buf.name = "mask.png"
            return buf
        
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        images = []
        if image1:
            images.append(base64_to_file(image1))
        if image2:
            images.append(base64_to_file(image2))

        request_args = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "image": images,
            "quality": quality,
            "n": n,
        }
        if model != "gpt-image-2":
            request_args["input_fidelity"] = input_fidelity
        if mask:
            request_args["mask"] = mask_b64_to_file(mask)

        response = client.images.edit(
            **request_args
        )

        image_results = [
            item.b64_json
            for item in response.data
            if getattr(item, "b64_json", None)
        ]

        if image_results:
            output_t = torch.cat(
                [pil2tensor(base64_to_image(image_base64)) for image_base64 in image_results],
                dim=0,
            )
        else:
            output_t = empty_image()

        result = str(response)
        return (output_t, result)