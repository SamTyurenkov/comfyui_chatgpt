from .image_generation import ChatGPTImageGenerationNode
from .image_generation import ChatGPTImageEditNode
from .image_generation import ImageToBase64

NODE_CLASS_MAPPINGS = {
    "ChatGPTImageGenerationNode": ChatGPTImageGenerationNode,
    "ChatGPTImageEditNode": ChatGPTImageEditNode,
    "ImageToBase64": ImageToBase64
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPTImageGenerationNode": "Chat GPT Image Generation Node",
    "ChatGPTImageEditNode": "Chat GPT Image Edit Node",
    "ImageToBase64": "Convert Image to Base64"
}