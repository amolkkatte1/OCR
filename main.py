import os
import json
import re
import io

from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from PIL import Image, ImageOps

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# DO NOT CHANGE
MODEL_NAME = "gemini-3.8-flash"

# Low thinking = lower cost + lower latency
THINKING_LEVEL = "low"

# ============================================================
# COST OPTIMIZATION
# ============================================================

# IMPORTANT:
# Do not send 2500px images to Gemini.
#
# 768 is a good starting point for analyzer reports because
# most analyzer text is relatively large and structured.
#
# If accuracy drops, test 896 or 1024.
MAX_IMAGE_DIMENSION = 768

# JPEG target is mainly for network/memory optimization.
# Gemini token cost depends much more on image resolution
# than raw JPEG KB.
MAX_IMAGE_SIZE = 150 * 1024  # 150 KB

JPEG_START_QUALITY = 82
JPEG_MIN_QUALITY = 60
JPEG_QUALITY_STEP = 5


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gemini OCR API",
    version="1.2.0"
)


# ============================================================
# RESPONSE MODEL
# ============================================================

class OCRResponse(BaseModel):
    status: str
    data: Dict[str, str]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "SUCCESS",
        "message": "Gemini OCR API is running"
    }


# ============================================================
# KEY NORMALIZATION
# ============================================================

def normalize_key(key: str) -> str:
    """
    Used only for comparing Gemini response keys
    with requested keys.

    Examples:

    LYMPH%  -> LYMPH
    LYMPH % -> LYMPH
    RDW-CV  -> RDWCV
    RDW CV  -> RDWCV
    """

    if key is None:
        return ""

    key = str(key).strip().upper()

    key = key.replace(" ", "")
    key = key.replace("-", "")
    key = key.replace("_", "")
    key = key.replace("%", "")

    return key


# ============================================================
# PARSE KEYS
# ============================================================

def parse_keys(keys: str) -> List[str]:
    """
    Supports:

    ["WBC","HGB","RBC"]

    and:

    [["WBC","HGB","RBC"]]
    """

    try:

        parsed = json.loads(keys)

    except Exception as e:

        raise ValueError(
            f"Invalid keys JSON: {str(e)}"
        )

    # Handle nested list
    if (
        isinstance(parsed, list)
        and len(parsed) > 0
        and isinstance(parsed[0], list)
    ):

        parsed = parsed[0]

    if not isinstance(parsed, list):

        raise ValueError(
            "keys must be a JSON array"
        )

    result = []

    for key in parsed:

        if key is None:
            continue

        key = str(key).strip()

        if key:
            result.append(key)

    if not result:

        raise ValueError(
            "No valid keys supplied"
        )

    return result


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_bytes: bytes,
    content_type: str
) -> tuple[bytes, str]:

    """
    Cost-optimized image preprocessing.

    IMPORTANT:

    Every image is normalized to a maximum dimension of 768px.

    This is intentional because sending a large image to Gemini
    can increase image input tokens and therefore API cost.

    Rules:

    1. Open image.
    2. Correct EXIF orientation.
    3. Convert to RGB.
    4. Resize proportionally to max 768px.
    5. Compress as JPEG.
    6. Try to keep image <= 150 KB.
    7. Never crop.
    """

    original_size = len(image_bytes)

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

    except Exception as e:

        raise ValueError(
            f"Invalid image: {str(e)}"
        )

    original_width, original_height = image.size

    print()
    print("=" * 70)
    print("IMAGE INFORMATION")
    print("=" * 70)

    print(
        "Original size:",
        original_size,
        "bytes"
    )

    print(
        "Original dimensions:",
        original_width,
        "x",
        original_height
    )

    print(
        "Original MIME:",
        content_type
    )

    # --------------------------------------------------------
    # Correct orientation FIRST
    # --------------------------------------------------------

    try:

        image = ImageOps.exif_transpose(
            image
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    if image.mode != "RGB":

        if image.mode in ("RGBA", "LA"):

            background = Image.new(
                "RGB",
                image.size,
                "white"
            )

            alpha = image.getchannel("A")

            background.paste(
                image,
                mask=alpha
            )

            image = background

        else:

            image = image.convert("RGB")

    else:

        image = image.copy()

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    width, height = image.size

    max_dimension = max(
        width,
        height
    )

    if max_dimension > MAX_IMAGE_DIMENSION:

        scale = (
            MAX_IMAGE_DIMENSION
            / max_dimension
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        print(
            "Resizing:",
            f"{width}x{height}",
            "->",
            f"{new_width}x{new_height}"
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    else:

        print(
            "Resize:",
            "not required"
        )

    # --------------------------------------------------------
    # JPEG compression
    # --------------------------------------------------------

    best_bytes = None
    best_quality = JPEG_START_QUALITY

    quality = JPEG_START_QUALITY

    while quality >= JPEG_MIN_QUALITY:

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=False
        )

        compressed_bytes = output.getvalue()

        print(
            "JPEG quality:",
            quality,
            "size:",
            len(compressed_bytes),
            "bytes"
        )

        best_bytes = compressed_bytes
        best_quality = quality

        if len(compressed_bytes) <= MAX_IMAGE_SIZE:

            break

        quality -= JPEG_QUALITY_STEP

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print(
        "Final image:",
        len(best_bytes),
        "bytes"
    )

    print(
        "Final JPEG quality:",
        best_quality
    )

    print(
        "Final dimensions:",
        image.size[0],
        "x",
        image.size[1]
    )

    print(
        "Estimated size reduction:",
        f"{(1 - len(best_bytes) / original_size) * 100:.1f}%"
    )

    print("=" * 70)

    return best_bytes, "image/jpeg"


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_json_response(
    text: str
) -> Dict[str, Any]:

    if not text:

        raise ValueError(
            "Gemini returned an empty response"
        )

    text = text.strip()

    # Remove markdown code fence
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Extract JSON object
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        json_text = match.group(0)

        try:

            parsed = json.loads(
                json_text
            )

            if isinstance(parsed, dict):

                return parsed

        except json.JSONDecodeError as e:

            raise ValueError(
                "Invalid JSON returned by Gemini: "
                f"{str(e)}"
            )

    raise ValueError(
        "Gemini response does not contain "
        "a valid JSON object"
    )


# ============================================================
# FIND VALUE FOR REQUESTED KEY
# ============================================================

def find_value_for_key(
    gemini_data: Dict[str, Any],
    requested_key: str
) -> str:

    requested_normalized = normalize_key(
        requested_key
    )

    for response_key, value in gemini_data.items():

        response_normalized = normalize_key(
            str(response_key)
        )

        if response_normalized == requested_normalized:

            if value is None:
                return ""

            return str(value).strip()

    return ""


# ============================================================
# CREATE GEMINI PROMPT
# ============================================================

def create_prompt(
    requested_keys: List[str],
    column_count: int
) -> str:

    keys_json = json.dumps(
        requested_keys,
        ensure_ascii=False
    )

    # IMPORTANT:
    # Keep this prompt short.
    #
    # The old prompt repeated many instructions and examples.
    # This version preserves the important OCR rules while
    # reducing unnecessary input tokens.

    prompt = f"""
OCR the laboratory analyzer image.

Requested keys:
{keys_json}

Column count:
{column_count}

Rules:

1. Extract ONLY the requested keys.
2. Use physical image position to match each key with its nearby value.
3. Read top-to-bottom and left-to-right.
4. For multiple columns, never take a value from another column.
5. Ignore spaces, hyphens and underscores when matching keys.
6. Treat LYMPH% and LYMPH % as the same key.
7. Treat RDW-CV and RDW CV as the same key.
8. Ignore H, L, HIGH, LOW and * flags between key and value.
9. Return only the actual result value.
10. Do not return units or reference ranges.
11. Never guess.
12. If a requested key cannot be confidently found, return "".
13. Return every requested key exactly as supplied.
14. Do not add extra keys.
15. Return ONLY valid JSON.

Example:
WBC 5.2 4.0-10.0
Return:
{{"WBC":"5.2"}}

For multiple columns:
WBC 5.2    HGB 10.7
Keep WBC=5.2 and HGB=10.7.

Return one JSON object containing all requested keys.
"""

    return prompt


# ============================================================
# PRINT TOKEN USAGE
# ============================================================

def print_usage(response):

    print()
    print("=" * 70)
    print("GEMINI TOKEN USAGE")
    print("=" * 70)

    usage = getattr(
        response,
        "usage_metadata",
        None
    )

    if usage is None:

        print(
            "Usage metadata not available."
        )

        print("=" * 70)

        return

    prompt_tokens = getattr(
        usage,
        "prompt_token_count",
        None
    )

    output_tokens = getattr(
        usage,
        "candidates_token_count",
        None
    )

    thinking_tokens = getattr(
        usage,
        "thoughts_token_count",
        None
    )

    total_tokens = getattr(
        usage,
        "total_token_count",
        None
    )

    print(
        "Input tokens:",
        prompt_tokens
    )

    print(
        "Output tokens:",
        output_tokens
    )

    print(
        "Thinking tokens:",
        thinking_tokens
    )

    print(
        "Total tokens:",
        total_tokens
    )

    print("=" * 70)


# ============================================================
# OCR ENDPOINT
# ============================================================

@app.post(
    "/extract",
    response_model=OCRResponse
)
async def extract(
    image: UploadFile = File(...),
    keys: str = Form(...),
    columnCount: int = Form(...)
):

    # ========================================================
    # Validate columnCount
    # ========================================================

    if columnCount <= 0:

        print(
            "ERROR: columnCount must be greater than 0"
        )

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # Parse keys
    # ========================================================

    try:

        requested_keys = parse_keys(
            keys
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("KEY PARSING ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # NEW REQUEST
    # ========================================================

    print()
    print("=" * 70)
    print("NEW OCR REQUEST")
    print("=" * 70)

    print(
        "Requested keys:",
        requested_keys
    )

    print(
        "Column count:",
        columnCount
    )

    # ========================================================
    # Validate MIME type
    # ========================================================

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed_types:

        print(
            "ERROR: Unsupported image type:",
            image.content_type
        )

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # Read image
    # ========================================================

    try:

        original_image_bytes = await image.read()

    except Exception as e:

        print()
        print("=" * 70)
        print("IMAGE READ ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    if not original_image_bytes:

        print(
            "ERROR: Empty image"
        )

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    try:

        image_bytes, image_mime_type = (
            preprocess_image(
                original_image_bytes,
                image.content_type
            )
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("IMAGE PROCESSING ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # CREATE PROMPT
    # ========================================================

    prompt = create_prompt(
        requested_keys,
        columnCount
    )

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        print()
        print("=" * 70)
        print("CALLING GEMINI")
        print("=" * 70)

        print(
            "Model:",
            MODEL_NAME
        )

        print(
            "Thinking:",
            THINKING_LEVEL
        )

        print(
            "Image sent:",
            len(image_bytes),
            "bytes"
        )

        print(
            "Image MIME:",
            image_mime_type
        )

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[

                types.Part.from_text(
                    text=prompt
                ),

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_mime_type
                )
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json",

                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL
                )
            )
        )

        print()
        print(
            "GEMINI RESPONSE RECEIVED"
        )

        # ====================================================
        # TOKEN USAGE
        # ====================================================

        print_usage(
            response
        )

        # ====================================================
        # RESPONSE TEXT
        # ====================================================

        response_text = response.text

        print()
        print(
            "Gemini response:"
        )

        print(
            response_text
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("GEMINI ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # PARSE GEMINI JSON
    # ========================================================

    try:

        gemini_data = clean_json_response(
            response_text
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("JSON PARSING ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print()
        print(
            "Gemini raw response:"
        )

        print(
            response_text
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # ========================================================
    # BUILD FINAL RESPONSE
    # ========================================================

    final_data = {}

    for requested_key in requested_keys:

        value = find_value_for_key(
            gemini_data,
            requested_key
        )

        final_data[
            requested_key
        ] = value

    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL OCR RESULT")
    print("=" * 70)

    print(
        json.dumps(
            final_data,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 70)

    # ========================================================
    # SUCCESS
    # ========================================================

    return OCRResponse(
        status="SUCCESS",
        data=final_data
    )
