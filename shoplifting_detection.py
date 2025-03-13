import os
from groq import Groq
import instructor
from pydantic import BaseModel,Field
from typing import Literal
import os
import base64
import io
from PIL import Image
from groq import Groq
import time
# Set Groq API key
class Classifier(BaseModel):
  Label:Literal["Shoplifting","Normal"]=Field(description="Label of the frame of boy standing")
  Confidence:float=Field(description="Confidence of the label")
  

os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
client=instructor.from_groq(client)
# Load image (FIX: Ensure correct color format)




# Convert BGR to RGB (FIX: OpenCV loads images in BGR format)

# Encode image to Base64
def encode_numpy_image(image_array):
    """Convert a NumPy image array to a Base64-encoded string."""
    image = Image.fromarray(image_array)  # Convert to 8-bit image
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Save as JPEG (or PNG)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Create chat request
def classifier(image):
    try:
      chat_completion = client.chat.completions.create(
          messages=[
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": "Based on the given image classify with confidence"},
                      {
                          "type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{encode_numpy_image(image)}"},
                      },
                  ],
              },
          ],
          temperature=0,
          model="llama-3.2-11b-vision-preview",
          response_model=Classifier,
      )
    except Exception as e:
      return "Normal",0.5
    time.sleep(1)
    # Print the model's response
    return chat_completion.Label,float(chat_completion.Confidence)
