
import base64
from pathlib import Path
from langchain_groq import ChatGroq
import config

def encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_media_type(image_path: str) -> str:
    """Return the MIME type for the given image file."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mime_map.get(ext, "image/jpeg")

def create_llm() -> ChatGroq:
    """Instantiate the Groq vision LLM using config settings."""
    return ChatGroq(model_name=config.MODEL_NAME, temperature=0)
