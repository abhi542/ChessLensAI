
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Application Config
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
OUTPUT_DIR = Path("output")

# Tracing
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ChessSheetOCR")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in environment variables.")
