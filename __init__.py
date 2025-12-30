from .image_generation import ChatGPTImageGenerationNode
from .image_generation import ChatGPTImageEditNode

from .banana_generation import BananaImageGenerationNode
from .banana_generation import BananaImageEditNode

from .utils import ImageToBase64

NODE_CLASS_MAPPINGS = {
    "ChatGPTImageGenerationNode": ChatGPTImageGenerationNode,
    "ChatGPTImageEditNode": ChatGPTImageEditNode,
    "BananaImageGenerationNode": BananaImageGenerationNode,
    "BananaImageEditNode": BananaImageEditNode,
    "ImageToBase64": ImageToBase64
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPTImageGenerationNode": "Chat GPT Image Generation Node",
    "ChatGPTImageEditNode": "Chat GPT Image Edit Node",
    "BananaImageGenerationNode": "Banana Image Generation Node",
    "BananaImageEditNode": "Banana Image Edit Node",
    "ImageToBase64": "Convert Image to Base64"
}