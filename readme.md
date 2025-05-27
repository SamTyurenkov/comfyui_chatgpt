Chat GPT Image Generation Chat Node for Comfy UI

1) Requires OPENAI_API_KEY to be available in the environment variables either through .env file or passed through CI
2) Supports two modes:   
--Start dialogue - can add up to 2 images to the request
--Continue dialogue - requires previous response ID to be added, load image nodes are ignored in that case.