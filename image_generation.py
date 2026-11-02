import base64
import os
from io import BytesIO
from PIL import Image, ImageOps
import numpy
import torch
from dotenv import load_dotenv
from .utils import ImageToBase64, OpenAIClient

load_dotenv()

# https://pypi.org/project/openai/

# New widgets go at the end of "optional", after seed. Saved workflows store widget
# values by position, so inserting a widget earlier shifts every value after it.

def b64_images_to_tensors(images_base64):
    """Decode base64 images into an IMAGE batch and a MASK batch from their alpha
    (1 = transparent, like LoadImage). A black 1x1 image when there are none."""
    images, masks = [], []
    for image_base64 in images_base64:
        image = ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(image_base64))))
        if "A" in image.getbands():
            alpha = numpy.array(image.getchannel("A")).astype(numpy.float32) / 255.0
        else:
            alpha = numpy.ones((image.height, image.width), dtype=numpy.float32)
        images.append(torch.from_numpy(numpy.array(image.convert("RGB")).astype(numpy.float32) / 255.0).unsqueeze(0))
        masks.append(torch.from_numpy(1.0 - alpha).unsqueeze(0))
    if not images:
        return torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1))
    return torch.cat(images, dim=0), torch.cat(masks, dim=0)


class ChatGPTImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high", "xhigh", "max"]
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
        IMAGE_TOOL_IMAGE_MODELS = [
            "auto",
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-1-mini",
        ]
        BACKGROUND_MODES = ["auto", "opaque", "transparent"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image based on provided image(s).", "multiline": True}),
                "model": (IMAGE_TOOL_MODELS, {"default": "gpt-4.1-mini"}),
                "size": ("STRING", {"default": "auto"}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "response_id": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image_model": (IMAGE_TOOL_IMAGE_MODELS, {"default": "auto"}),
                "background": (BACKGROUND_MODES, {"default":"auto"})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "MASK")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, moderation, quality, image1=None, image2=None, response_id=None, seed=0, image_model="auto", background="auto"):

        client = OpenAIClient(
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

        image_tool = {
            "type": "image_generation",
            "size": size,
            "moderation": moderation,
            "quality": quality,
            "background": background,
        }
        if image_model and image_model != "auto":
            image_tool["model"] = image_model

        request_args = {
            "model": model,
            "input":messages,
            "tools":[image_tool]
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

        output_t, mask_t = b64_images_to_tensors(image_results)

        result = str(response)
        return (output_t, result, response.id, mask_t)


class ChatGPTImageModelGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high", "xhigh", "max"]
        BACKGROUND_MODES = ["auto", "opaque", "transparent"]
        OUTPUT_FORMATS = ["png", "jpeg", "webp"]
        IMAGE_MODELS = [
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-1-mini",
        ]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image according to this prompt.", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2.5-sunburst"}),
                "size": ("STRING", {"default": "auto"}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "background": (BACKGROUND_MODES, {"default":"auto"}),
                "output_format": (OUTPUT_FORMATS, {"default":"png"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, moderation, quality, n, seed=0, background="auto", output_format="png"):

        client = OpenAIClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        request_args = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "moderation": moderation,
            "quality": quality,
            "background": background,
            "output_format": output_format,
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

        output_t, mask_t = b64_images_to_tensors(image_results)

        result = str(response)
        return (output_t, result, mask_t)


class ChatGPTImageEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODERATION_MODES = ["auto", "low"]
        QUALITY_MODES = ["auto", "low", "medium", "high", "xhigh", "max"]
        INPUT_FIDELITY_MODES = ["high", "low"]
        BACKGROUND_MODES = ["auto", "opaque", "transparent"]
        OUTPUT_FORMATS = ["png", "jpeg", "webp"]
        IMAGE_MODELS = [
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-1-mini",
        ]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit image according to this prompt.", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2.5-sunburst"}),
                "size": ("STRING", {"default": "auto"}),
                "quality": (QUALITY_MODES, {"default":"auto"}),
                "input_fidelity": (INPUT_FIDELITY_MODES, {"default":"low"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "image3": ("STRING",),
                "image4": ("STRING",),
                "mask": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "moderation": (MODERATION_MODES, {"default":"auto"}),
                "background": (BACKGROUND_MODES, {"default":"auto"}),
                "output_format": (OUTPUT_FORMATS, {"default":"png"})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, model, size, quality, input_fidelity, n, image1=None, image2=None, image3=None, image4=None, mask=None, seed=0, moderation="auto", background="auto", output_format="png"):

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

        client = OpenAIClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        images = []
        if image1:
            images.append(base64_to_file(image1))
        if image2:
            images.append(base64_to_file(image2))
        if image3:
            images.append(base64_to_file(image3))
        if image4:
            images.append(base64_to_file(image4))

        request_args = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "image": images,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "n": n,
            # The openai SDK has no moderation argument on images.edit, the API accepts it.
            "extra_body": {"moderation": moderation},
        }
        if not model.startswith("gpt-image-2"):
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

        output_t, mask_t = b64_images_to_tensors(image_results)

        result = str(response)
        return (output_t, result, mask_t)
