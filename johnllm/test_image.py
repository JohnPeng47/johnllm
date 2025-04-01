import base64
import openai
from pydantic import BaseModel

from src.core import LLMModel

client = openai.OpenAI()

def read_image_as_b64(image_path: str) -> str:
    """Read an image file and return its base64 encoded string representation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")
        return base64_string

# message = [{
#     "role": "user",
#     "content": [
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{read_image_as_b64('bamler.jpeg')}",
#             },
#         },
#         {
#             "type": "text",
#             "text": "Describe the contents of this image"
#         }
#     ]
# }]

message = [{
    "role": "user",
    "content": "What is the capital of France?"
}]

model = LLMModel()
res = model.invoke(message, model_name="gpt-4o", response_format=None)
print(res)