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


# CONFIG
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.8-flash"
THINKING_LEVEL = "low"

MAX_IMAGE_DIMENSION = 768
MAX_IMAGE_SIZE = 150 * 1024
JPEG_START_QUALITY = 82
JPEG_MIN_QUALITY = 60
JPEG_QUALITY_STEP = 5

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Gemini OCR API",
    version="1.2.0"
)


class OCRResponse(BaseModel):
    status: str
    data: Dict[str, str]


@app.get("/health")
def health():
    return {
        "status": "SUCCESS",
        "message": "Gemini OCR API is running"
    }


def normalize_key(key: str) -> str:
    if key is None:
        return ""

    key = str(key).strip().upper()
    key = key.replace(" ", "")
    key = key.replace("-", "")
    key = key.replace("_", "")
    key = key.replace("%", "")

    return key


def parse_keys(keys: str) -> List[str]:

    try:
        parsed = json.loads(keys)

    except Exception as e:
        raise ValueError(
            f"Invalid keys JSON: {str(e)}"
        )

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


def preprocess_image(
    image_bytes: bytes,
    original_mime_type: str
):

    print()
    print("=" * 70)
    print("IMAGE INFORMATION")
    print("=" * 70)

    print(
        "Original size:",
        len(image_bytes),
        "bytes"
    )

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image.load()

    original_width, original_height = image.size

    print(
        "Original dimensions:",
        f"{original_width}x{original_height}"
    )

    print(
        "Original MIME:",
        original_mime_type
    )

    # Fix EXIF orientation
    image = ImageOps.exif_transpose(image)

    # Convert image to RGB
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

    elif image.mode != "RGB":

        image = image.convert("RGB")

    # Resize proportionally
    max_dimension = max(
        image.width,
        image.height
    )

    if max_dimension > MAX_IMAGE_DIMENSION:

        scale = (
            MAX_IMAGE_DIMENSION
            / max_dimension
        )

        new_width = max(
            1,
            int(image.width * scale)
        )

        new_height = max(
            1,
            int(image.height * scale)
        )

        print(
            "Resizing:",
            f"{image.width}x{image.height}",
            "->",
            f"{new_width}x{new_height}"
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    # JPEG compression
    quality = JPEG_START_QUALITY
    output_bytes = None

    while quality >= JPEG_MIN_QUALITY:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=False
        )

        candidate = buffer.getvalue()

        if (
            len(candidate)
            <= MAX_IMAGE_SIZE
        ):
            output_bytes = candidate
            break

        quality -= JPEG_QUALITY_STEP

    if output_bytes is None:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=JPEG_MIN_QUALITY,
            optimize=True,
            progressive=False
        )

        output_bytes = buffer.getvalue()

    final_size = len(output_bytes)

    print(
        "Final size:",
        final_size,
        "bytes"
    )

    print(
        "Final quality:",
        quality
        if quality >= JPEG_MIN_QUALITY
        else JPEG_MIN_QUALITY
    )

    print(
        "Final dimensions:",
        f"{image.width}x{image.height}"
    )

    reduction = (
        (1 - final_size / len(image_bytes))
        * 100
    )

    print(
        "Size reduction:",
        f"{reduction:.2f}%"
    )

    print("=" * 70)

    return output_bytes, "image/jpeg"


def clean_json_response(
    text: str
) -> Dict[str, Any]:

    text = text.strip()

    # Remove markdown JSON fences
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

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # Try extracting JSON object
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

        except Exception:
            pass

    raise ValueError(
        "Invalid JSON response from Gemini"
    )


def find_value_for_key(
    gemini_data: Dict[str, Any],
    requested_key: str
) -> str:

    normalized_requested = normalize_key(
        requested_key
    )

    for response_key, response_value in gemini_data.items():

        normalized_response = normalize_key(
            response_key
        )

        if (
            normalized_response
            == normalized_requested
        ):

            if response_value is None:
                return ""

            return str(
                response_value
            ).strip()

    return ""


def convert_result_value(
    requested_key: str,
    value: str
) -> str:

    if not value:
        return value

    normalized_key = normalize_key(
        requested_key
    )

    # WBC
    # Example:
    # 11.0 -> 11000
    # 5.7  -> 5700
    if normalized_key == "WBC":

        try:

            number = float(value)

            converted = number * 1000

            if converted.is_integer():

                return str(
                    int(converted)
                )

            return str(converted)

        except (
            ValueError,
            TypeError
        ):

            return value

    # PLT
    # Example:
    # 232 -> 232000
    # 104 -> 104000
    # 84  -> 84000
    if normalized_key == "PLT":

        try:

            number = float(value)

            converted = number * 1000

            if converted.is_integer():

                return str(
                    int(converted)
                )

            return str(converted)

        except (
            ValueError,
            TypeError
        ):

            return value

    # LYMPH% and GRAN%
    # Only take the integer part.
    #
    # 32.6 -> 32
    # 32.4 -> 32
    # 58.7 -> 58
    # 41.2 -> 41
    #
    # normalize_key() removes %,
    # therefore:
    # LYMPH% -> LYMPH
    # GRAN%  -> GRAN
    if normalized_key in (
        "LYMPH",
        "GRAN"
    ):

        try:

            number = float(value)

            return str(
                int(number)
            )

        except (
            ValueError,
            TypeError
        ):

            return value

    # All other keys remain unchanged
    return value


def create_prompt(
    requested_keys: List[str],
    column_count: int
) -> str:

    keys_json = json.dumps(
        requested_keys,
        ensure_ascii=False
    )

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


@app.post(
    "/extract",
    response_model=OCRResponse
)
async def extract(
    image: UploadFile = File(...),
    keys: str = Form(...),
    columnCount: int = Form(...)
):

    print()
    print("=" * 70)
    print("NEW OCR REQUEST")
    print("=" * 70)

    try:

        # Validate column count
        if columnCount <= 0:

            return OCRResponse(
                status="ERROR",
                data={}
            )

        # Parse requested keys dynamically
        requested_keys = parse_keys(
            keys
        )

        print(
            "Requested keys:",
            requested_keys
        )

        print(
            "Column count:",
            columnCount
        )

        # Validate image MIME
        allowed_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp"
        }

        if image.content_type not in allowed_types:

            print(
                "Unsupported image type:",
                image.content_type
            )

            return OCRResponse(
                status="ERROR",
                data={}
            )

        # Read uploaded image
        original_image_bytes = await image.read()

        if not original_image_bytes:

            print(
                "Empty image received"
            )

            return OCRResponse(
                status="ERROR",
                data={}
            )

        # Preprocess image
        image_bytes, image_mime_type = (
            preprocess_image(
                original_image_bytes,
                image.content_type
            )
        )

        # Create dynamic prompt
        prompt = create_prompt(
            requested_keys,
            columnCount
        )

        print()
        print("=" * 70)
        print("SENDING REQUEST TO GEMINI")
        print("=" * 70)

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

        # Print Gemini token usage
        print_usage(
            response
        )

        # Raw response
        print()
        print("=" * 70)
        print("RAW GEMINI RESPONSE")
        print("=" * 70)

        print(
            response.text
        )

        print("=" * 70)

        # Parse Gemini JSON
        gemini_data = clean_json_response(
            response.text
        )

        # Build final response ONLY
        # using requested keys
        final_data = {}

        for requested_key in requested_keys:

            value = find_value_for_key(
                gemini_data,
                requested_key
            )

            # Apply local conversions
            value = convert_result_value(
                requested_key,
                value
            )

            final_data[
                requested_key
            ] = value

        print()
        print("=" * 70)
        print("FINAL OCR RESULT")
        print("=" * 70)

        print(
            json.dumps(
                final_data,
                ensure_ascii=False
            )
        )

        print("=" * 70)

        return OCRResponse(
            status="SUCCESS",
            data=final_data
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("GEMINI ERROR")
        print("=" * 70)

        print(
            "Error:",
            str(e)
        )

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )
