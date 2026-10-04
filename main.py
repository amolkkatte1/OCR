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

MODEL_NAME = "gemini-3.8-flash"

# Low thinking = lower cost + lower latency
# Gemini 3.8 Flash supports: low, medium, high
THINKING_LEVEL = "low"

# Maximum dimension sent to Gemini
MAX_IMAGE_DIMENSION = 2500

# Target maximum image size
MAX_IMAGE_SIZE = 500 * 1024  # 500 KB

# JPEG quality range
JPEG_START_QUALITY = 85
JPEG_MIN_QUALITY = 55
JPEG_QUALITY_STEP = 5


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gemini OCR API",
    version="1.1.0"
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
    Normalize keys only for comparison.

    Examples:

    LYMPH%   -> LYMPH
    LYMPH %  -> LYMPH
    RDW-CV   -> RDWCV
    RDW CV   -> RDWCV
    """

    if key is None:
        return ""

    key = str(key).strip().upper()

    # Remove spaces
    key = key.replace(" ", "")

    # Remove hyphens
    key = key.replace("-", "")

    # Remove underscores
    key = key.replace("_", "")

    # Keep % out of comparison
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

    # Nested list
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
    Image rules:

    1. If image <= 500 KB AND dimensions <= 2500:
       use original image.

    2. Otherwise:
       - convert to RGB
       - resize proportionally if dimension > 2500
       - compress as JPEG
       - try to keep <= 500 KB

    Returns:

        processed_bytes
        mime_type
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

    width, height = image.size

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
        width,
        "x",
        height
    )

    print(
        "Original MIME:",
        content_type
    )

    # --------------------------------------------------------
    # Rule 1:
    # Original image is acceptable
    # --------------------------------------------------------

    if (
        original_size <= MAX_IMAGE_SIZE
        and width <= MAX_IMAGE_DIMENSION
        and height <= MAX_IMAGE_DIMENSION
    ):

        print(
            "Image processing:",
            "NOT REQUIRED"
        )

        print(
            "Using original image."
        )

        print("=" * 70)

        return image_bytes, content_type

    # --------------------------------------------------------
    # Processing required
    # --------------------------------------------------------

    print(
        "Image processing:",
        "REQUIRED"
    )

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    if image.mode != "RGB":

        # Handle transparency correctly
        if image.mode in (
            "RGBA",
            "LA"
        ):

            background = Image.new(
                "RGB",
                image.size,
                "white"
            )

            if image.mode == "RGBA":

                background.paste(
                    image,
                    mask=image.getchannel("A")
                )

            else:

                background.paste(
                    image,
                    mask=image.getchannel("A")
                )

            image = background

        else:

            image = image.convert("RGB")

    else:

        image = image.copy()

    # --------------------------------------------------------
    # Correct image orientation
    # --------------------------------------------------------

    try:

        image = ImageOps.exif_transpose(
            image
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Resize proportionally
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
            progressive=True
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

    print("=" * 70)

    return best_bytes, "image/jpeg"


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_json_response(
    text: str
) -> Dict[str, Any]:

    """
    Convert Gemini response into JSON object.

    Handles:

    {
        "WBC": "5.2"
    }

    and markdown:

    ```json
    {
        "WBC": "5.2"
    }
    """

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

        if (
            response_normalized
            == requested_normalized
        ):

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

    prompt = f"""
You are a high-accuracy OCR extraction engine for
pathology and laboratory analyzer reports.

Extract ONLY the requested keys from the image.

REQUESTED KEYS:
{keys_json}

COLUMN COUNT:
{column_count}

LAYOUT:

The image may contain multiple key-value columns
on the same physical row.

columnCount={column_count} means a physical row
can contain up to {column_count} key-value pairs.

Example:

WBC  5.2        HGB  10.7

For columnCount=2:

WBC -> 5.2
HGB -> 10.7

Use the physical position of text in the image.
Do NOT match a value from another column.

Read:
1. top to bottom
2. left to right within each row
3. match each key with its nearby value

KEY MATCHING:

Ignore spaces, hyphens and underscores when
matching keys.

LYMPH% = LYMPH %
RDW-CV = RDW CV

Do NOT confuse RDW-CV with RDW-CV%.

H/L flags may appear between a key and its value.

Examples:

WBC 5.2
WBC H 5.2
WBC L 5.2

All mean:

WBC = "5.2"

Do not include H, L, HIGH, LOW or * as part
of the result.

VALUE:

Return only the actual result value.

Example:

WBC 5.2 4.0-10.0

Return:

"WBC": "5.2"

Do NOT return the reference range.

Do NOT return units.

Do NOT guess.

If a requested key cannot be confidently found,
return an empty string.

OUTPUT:

Return ONLY one valid JSON object.

Return every requested key exactly as provided.

Do not add extra keys.

Example:

{{
    "WBC": "5.2",
    "HGB": "10.7",
    "RBC": "4.03"
}}

Missing:

{{
    "WBC": "5.2",
    "HGB": "",
    "RBC": "4.03"
}}
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

    # Different SDK versions can expose
    # slightly different fields.

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
    # Preprocess image
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
    # Create prompt
    # ========================================================

    prompt = create_prompt(
        requested_keys,
        columnCount
    )

    # ========================================================
    # Call Gemini
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

        # ----------------------------------------------------
        # Print token usage
        # ----------------------------------------------------

        print_usage(
            response
        )

        # ----------------------------------------------------
        # Get response text
        # ----------------------------------------------------

        response_text = response.text

        print()
        print("Gemini response:")

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
    # Parse Gemini JSON
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
    # Build final response
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
    # Print final result
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


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

# Do NOT use reload=True in production.
#
# if __name__ == "__main__":
#
#     import uvicorn
#
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )

