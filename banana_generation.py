import base64
import os
from io import BytesIO
from PIL import Image
import numpy
from google import genai
from google.genai import types
import torch
from dotenv import load_dotenv
from .utils import ImageToBase64

load_dotenv()

# https://pypi.org/project/google-genai/

class BananaImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        ASPECT_RATIO_MODES = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
        RESOLUTION_MODES = ["1K", "2K", "4K"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate image based on provided image(s).", "multiline": True}),
                "aspect_ratio": (ASPECT_RATIO_MODES, {"default":"1:1"}),
                "resolution": (RESOLUTION_MODES, {"default":"2K"})
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

    def request(self, prompt, aspect_ratio, resolution, image1=None, image2=None, response_id=None, seed=0):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            # Ensure we have a PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array and then to tensor
            img_array = numpy.array(image).astype(numpy.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = Image.open(BytesIO(image_bytes))
            return image_data
        
        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
        )

        # Build contents list with prompt and images using types.Part.from_bytes()
        contents = []
        
        # Add images if provided (convert base64 to bytes and use types.Part.from_bytes)
        if image1 and not response_id:
            image_bytes = base64.b64decode(image1)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                )
            )
        
        if image2 and not response_id:
            image_bytes = base64.b64decode(image2)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                )
            )
        
        # Add text prompt
        contents.append(prompt)

        try:
            # Use gemini-3-pro-image-preview model with proper config
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    seed=seed,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    ),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                    ]
                )
            )
            
            # Extract image from response parts and convert full response to string
            output_image = None
            
            # Safely iterate over response parts to find image
            if hasattr(response, 'parts') and response.parts is not None:
                try:
                    for part in response.parts:
                        if part is None:
                            continue
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            # Get image data from inline_data and convert to PIL Image
                            inline_data = part.inline_data
                            if inline_data and hasattr(inline_data, 'data'):
                                image_bytes = inline_data.data
                                if image_bytes:
                                    # Convert bytes to PIL Image
                                    image = Image.open(BytesIO(image_bytes))
                                    # Ensure image is in RGB mode
                                    if image.mode != 'RGB':
                                        image = image.convert('RGB')
                                    output_image = image
                except Exception as parse_error:
                    # If parsing parts fails, will be included in full response string
                    pass
            
            # Convert entire response object to string for debugging
            try:
                # Try to get a detailed representation
                result = repr(response)
            except:
                try:
                    # Fallback to string conversion
                    result = str(response)
                except:
                    # Last resort - convert to dict if possible
                    try:
                        result = str(vars(response))
                    except:
                        result = f"Response object: {type(response)}"
            
            if output_image:
                output_t = pil2tensor(output_image)
                response_id_str = getattr(response, 'id', '') or ''
                return (output_t, result, response_id_str)
            else:
                # If no image found, return empty
                output_t = empty_image()
                response_id_str = getattr(response, 'id', '') or ''
                return (output_t, result, response_id_str)
            
        except Exception as e:
            # On error, return empty image
            output_t = empty_image()
            result = f"Error: {str(e)}"
            return (output_t, result, "")


class BananaImageEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        ASPECT_RATIO_MODES = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
        RESOLUTION_MODES = ["1K", "2K", "4K"]
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit image according to this prompt.", "multiline": True}),
                "aspect_ratio": (ASPECT_RATIO_MODES, {"default":"1:1"}),
                "resolution": (RESOLUTION_MODES, {"default":"2K"})
            },
            "optional": {
                "image1": ("STRING",),
                "image2": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "request"

    def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

    def request(self, prompt, aspect_ratio, resolution, image1=None, image2=None, seed=0):

        # Create a black 1x1 pixel image as placeholder
        def empty_image():
            img = Image.new("RGB", (1, 1))
            return pil2tensor(img)
    
        # Helper function to process an image tensor
        def pil2tensor(image):
            # Ensure we have a PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array and then to tensor
            img_array = numpy.array(image).astype(numpy.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        
        def base64_to_image(image_base64):
            image_bytes = base64.b64decode(image_base64)
            image_data = Image.open(BytesIO(image_bytes))
            return image_data

        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
        )

        # Build contents list with images using types.Part.from_bytes() and edit prompt
        contents = []
        
        if image1:
            image_bytes = base64.b64decode(image1)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                )
            )
        
        if image2:
            image_bytes = base64.b64decode(image2)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                )
            )
        
        # Add edit prompt
        contents.append(prompt)

        try:
            # Use gemini-3-pro-image-preview model for image editing
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    seed=seed,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    ),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                    ]
                )
            )
            
            # Extract image from response parts and convert full response to string
            output_image = None
            
            # Safely iterate over response parts to find image
            if hasattr(response, 'parts') and response.parts is not None:
                try:
                    for part in response.parts:
                        if part is None:
                            continue
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            # Get image data from inline_data and convert to PIL Image
                            inline_data = part.inline_data
                            if inline_data and hasattr(inline_data, 'data'):
                                image_bytes = inline_data.data
                                if image_bytes:
                                    # Convert bytes to PIL Image
                                    image = Image.open(BytesIO(image_bytes))
                                    # Ensure image is in RGB mode
                                    if image.mode != 'RGB':
                                        image = image.convert('RGB')
                                    output_image = image
                except Exception as parse_error:
                    # If parsing parts fails, will be included in full response string
                    pass
            
            # Convert entire response object to string for debugging
            try:
                # Try to get a detailed representation
                result = repr(response)
            except:
                try:
                    # Fallback to string conversion
                    result = str(response)
                except:
                    # Last resort - convert to dict if possible
                    try:
                        result = str(vars(response))
                    except:
                        result = f"Response object: {type(response)}"
            
            if output_image:
                output_t = pil2tensor(output_image)
                return (output_t, result)
            else:
                # If no image found, return empty
                output_t = empty_image()
                return (output_t, result)
            
        except Exception as e:
            # On error, return empty image
            output_t = empty_image()
            result = f"Error: {str(e)}"
            return (output_t, result)

