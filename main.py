import os
import io
import json
import logging
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, ImageOps
from google import genai
from google.genai import types
from pydantic import BaseModel

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

# Gemini model
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Maximum image dimension sent to Gemini.
# 768 is a good starting point for cost reduction.
MAX_IMAGE_DIMENSION = int(os.getenv("OCR_MAX_IMAGE_DIMENSION", "768"))

# Target JPEG size.
# This is only a local optimization; Gemini image-token cost is
# primarily affected by image resolution rather than raw JPEG KB.
TARGET_IMAGE_KB = int(os.getenv("OCR_TARGET_IMAGE_KB", "100"))

# JPEG quality range
JPEG_START_QUALITY = 82
JPEG_MIN_QUALITY = 60

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr")

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Pathology OCR API",
    version="2.0.0"
)

# ============================================================
# RESPONSE MODEL
# ============================================================

class OCRResponse(BaseModel):
    WBC: str
    LYMPH_percent: str = ""
    GRAN_percent: str = ""
    HGB: str
    RBC: str
    HCT: str
    RDW_CV: str
    PLT: str


# ============================================================
# REQUESTED KEYS
# ============================================================

REQUESTED_KEYS = [
    "WBC",
    "LYMPH%",
    "GRAN%",
    "HGB",
    "RBC",
    "HCT",
    "RDW-CV",
    "PLT"
]

# ============================================================
# GEMINI PROMPT
# ============================================================

OCR_PROMPT = """
You are a high-accuracy OCR engine for pathology/laboratory analyzer reports.

Extract ONLY these requested keys:
["WBC","LYMPH%","GRAN%","HGB","RBC","HCT","RDW-CV","PLT"]

columnCount=1.

Rules:
- Match each key to its nearby value using physical image position.
- Read top-to-bottom and left-to-right.
- Ignore spaces, hyphens and underscores when matching keys.
- LYMPH% = LYMPH %, RDW-CV = RDW CV.
- Ignore H/L/HIGH/LOW/* flags between key and value.
- Return only the actual result value, not units or reference ranges.
- Never guess. If not confidently found, return "".
- Return every requested key exactly as provided.
- Return ONLY valid JSON. No extra keys.

JSON format:
{
  "WBC": "",
  "LYMPH%": "",
  "GRAN%": "",
  "HGB": "",
  "RBC": "",
  "HCT": "",
  "RDW-CV": "",
  "PLT": ""
}
"""

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Resize and compress the image before sending it to Gemini.

    Important:
    - Keeps aspect ratio.
    - Converts to RGB.
    - Does not crop.
    - Does not aggressively sharpen.
    - Uses JPEG compression.
    """

    image = Image.open(io.BytesIO(image_bytes))

    logger.info(
        "Original image: %dx%d, %.2f KB",
        image.width,
        image.height,
        len(image_bytes) / 1024
    )

    # Fix camera EXIF orientation
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A") if "A" in image.getbands() else None

            if alpha:
                background.paste(image, mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")

    # --------------------------------------------------------
    # Resize only when needed
    # --------------------------------------------------------

    width, height = image.size
    max_dimension = max(width, height)

    if max_dimension > MAX_IMAGE_DIMENSION:

        scale = MAX_IMAGE_DIMENSION / max_dimension

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        logger.info(
            "Resizing image: %dx%d -> %dx%d",
            width,
            height,
            new_width,
            new_height
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # JPEG compression
    # --------------------------------------------------------

    quality = JPEG_START_QUALITY
    output = None

    while quality >= JPEG_MIN_QUALITY:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True
        )

        data = buffer.getvalue()
        size_kb = len(data) / 1024

        if size_kb <= TARGET_IMAGE_KB:
            output = data
            break

        quality -= 5

    if output is None:
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=JPEG_MIN_QUALITY,
            optimize=True,
            progressive=True
        )

        output = buffer.getvalue()

    logger.info(
        "Processed image: %dx%d, %.2f KB, quality=%d",
        image.width,
        image.height,
        len(output) / 1024,
        quality
    )

    return output


# ============================================================
# JSON CLEANING
# ============================================================

def clean_result(raw_text: str) -> dict:
    """
    Safely convert Gemini output into the exact requested JSON.
    """

    if not raw_text:
        return {key: "" for key in REQUESTED_KEYS}

    text = raw_text.strip()

    # Remove markdown JSON fences if Gemini returns them
    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```JSON", "", 1)

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

    try:
        data = json.loads(text)

    except Exception as e:
        logger.warning("Invalid Gemini JSON: %s", e)

        return {
            key: ""
            for key in REQUESTED_KEYS
        }

    # --------------------------------------------------------
    # Normalize possible alternative key formatting
    # --------------------------------------------------------

    normalized = {}

    for key, value in data.items():

        normalized_key = (
            str(key)
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .upper()
        )

        normalized[normalized_key] = value

    # --------------------------------------------------------
    # Map exact requested keys
    # --------------------------------------------------------

    result = {}

    aliases = {
        "WBC": ["WBC"],
        "LYMPH%": ["LYMPH%", "LYMPH"],
        "GRAN%": ["GRAN%", "GRAN"],
        "HGB": ["HGB"],
        "RBC": ["RBC"],
        "HCT": ["HCT"],
        "RDW-CV": ["RDWCV", "RDW-CV", "RDW"],
        "PLT": ["PLT"]
    }

    for requested_key in REQUESTED_KEYS:

        value = ""

        for alias in aliases[requested_key]:

            normalized_alias = (
                alias
                .replace(" ", "")
                .replace("-", "")
                .replace("_", "")
                .upper()
            )

            if normalized_alias in normalized:

                candidate = normalized[normalized_alias]

                if candidate is None:
                    value = ""
                elif isinstance(candidate, (dict, list)):
                    value = ""
                else:
                    value = str(candidate).strip()

                break

        result[requested_key] = value

    return result


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "UP",
        "model": MODEL_NAME,
        "maxImageDimension": MAX_IMAGE_DIMENSION,
        "targetImageKB": TARGET_IMAGE_KB
    }


# ============================================================
# OCR ENDPOINT
# ============================================================

@app.post("/ocr/extract")
async def extract_ocr(
    image: UploadFile = File(...),
    keys: str = Form(default='["WBC","LYMPH%","GRAN%","HGB","RBC","HCT","RDW-CV","PLT"]'),
    columnCount: str = Form(default="1")
):

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        original_bytes = await image.read()

        if not original_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image"
            )

        logger.info("=" * 70)
        logger.info("NEW OCR REQUEST")
        logger.info("Original upload: %.2f KB", len(original_bytes) / 1024)
        logger.info("Requested keys: %s", keys)
        logger.info("Column count: %s", columnCount)
        logger.info("=" * 70)

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_bytes = preprocess_image(original_bytes)

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(
                    data=processed_bytes,
                    mime_type="image/jpeg"
                ),
                OCR_PROMPT
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "WBC": {
                            "type": "string"
                        },
                        "LYMPH%": {
                            "type": "string"
                        },
                        "GRAN%": {
                            "type": "string"
                        },
                        "HGB": {
                            "type": "string"
                        },
                        "RBC": {
                            "type": "string"
                        },
                        "HCT": {
                            "type": "string"
                        },
                        "RDW-CV": {
                            "type": "string"
                        },
                        "PLT": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "WBC",
                        "LYMPH%",
                        "GRAN%",
                        "HGB",
                        "RBC",
                        "HCT",
                        "RDW-CV",
                        "PLT"
                    ]
                }
            )
        )

        # ----------------------------------------------------
        # Gemini result
        # ----------------------------------------------------

        raw_text = response.text

        logger.info("Gemini response: %s", raw_text)

        result = clean_result(raw_text)

        logger.info("Final OCR result: %s", result)

        return result

    except HTTPException:
        raise

    except Exception as e:

        logger.exception("OCR processing failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
