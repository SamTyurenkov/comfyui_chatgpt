import base64
from io import BytesIO

import numpy
from openai import AsyncOpenAI, OpenAI
from PIL import Image


class _SkipRateLimitRetryMixin:
    """Keep default retries for 408/409/5xx, but never retry 429."""

    def _should_retry(self, response):
        if getattr(response, "status_code", None) == 429:
            return False
        return super()._should_retry(response)


class OpenAIClient(_SkipRateLimitRetryMixin, OpenAI):
    pass


class AsyncOpenAIClient(_SkipRateLimitRetryMixin, AsyncOpenAI):
    pass


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
      def tensor2pil(image):
        return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))
      
      pil_image = tensor2pil(image)
      buffered = BytesIO()
      pil_image.save(buffered, format="PNG")
      image_bytes = buffered.getvalue()

      base64_str = base64.b64encode(image_bytes).decode("utf-8")
      return {"result": (base64_str,)}

