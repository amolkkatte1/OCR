import os
import io
import json
import logging

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set"
    )


# ============================================================
# GEMINI MODEL
# ============================================================

# DO NOT CHANGE
MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# IMAGE COST OPTIMIZATION
# ============================================================

# Maximum width/height sent to Gemini.
#
# 768  -> lower image token usage
# 1024 -> better safety for small text
#
# Default = 768
MAX_IMAGE_DIMENSION = int(
    os.getenv(
        "OCR_MAX_IMAGE_DIMENSION",
        "768"
    )
)


# Target JPEG size.
#
# This mainly reduces upload/network size.
# Resolution reduction is the more important
# part for vision-token cost.
TARGET_IMAGE_KB = int(
    os.getenv(
        "OCR_TARGET_IMAGE_KB",
        "100"
    )
)


JPEG_START_QUALITY = 82
JPEG_MIN_QUALITY = 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ocr")


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
    title="Pathology OCR API",
    version="2.0.0"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_bytes: bytes
) -> bytes:

    """
    Resize/compress image before Gemini.

    Steps:
    1. Read image
    2. Fix EXIF camera rotation
    3. Convert to RGB
    4. Resize if larger than MAX_IMAGE_DIMENSION
    5. Compress as JPEG
    6. Return bytes

    No cropping is performed.
    Aspect ratio is preserved.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    logger.info(
        "Original image: %dx%d, %.2f KB",
        image.width,
        image.height,
        len(image_bytes) / 1024
    )

    # --------------------------------------------------------
    # Fix camera orientation
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(
        image
    )

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    if image.mode != "RGB":

        if image.mode in (
            "RGBA",
            "LA"
        ):

            background = Image.new(
                "RGB",
                image.size,
                "white"
            )

            if "A" in image.getbands():

                alpha = image.getchannel(
                    "A"
                )

                background.paste(
                    image,
                    mask=alpha
                )

            else:

                background.paste(
                    image
                )

            image = background

        else:

            image = image.convert(
                "RGB"
            )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    width = image.width
    height = image.height

    largest_dimension = max(
        width,
        height
    )

    if largest_dimension > MAX_IMAGE_DIMENSION:

        scale = (
            MAX_IMAGE_DIMENSION /
            largest_dimension
        )

        new_width = max(
            1,
            round(width * scale)
        )

        new_height = max(
            1,
            round(height * scale)
        )

        logger.info(
            "Resizing image: %dx%d -> %dx%d",
            width,
            height,
            new_width,
            new_height
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # JPEG compression
    # --------------------------------------------------------

    processed_bytes = None
    selected_quality = JPEG_MIN_QUALITY

    for quality in range(
        JPEG_START_QUALITY,
        JPEG_MIN_QUALITY - 1,
        -5
    ):

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        data = buffer.getvalue()

        if len(data) <= (
            TARGET_IMAGE_KB * 1024
        ):

            processed_bytes = data
            selected_quality = quality
            break

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if processed_bytes is None:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=JPEG_MIN_QUALITY,
            optimize=True
        )

        processed_bytes = buffer.getvalue()

    logger.info(
        "Processed image: %dx%d, %.2f KB, quality=%d",
        image.width,
        image.height,
        len(processed_bytes) / 1024,
        selected_quality
    )

    return processed_bytes


# ============================================================
# PARSE KEYS
# ============================================================

def parse_keys(
    keys_input: str
) -> list[str]:

    """
    Accept keys from the API request.

    Supported:

    ["WBC","HGB","RBC"]

    or:

    [["WBC","HGB","RBC"]]

    or comma-separated:

    WBC,HGB,RBC
    """

    if not keys_input:
        return []

    # --------------------------------------------------------
    # Try JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(
            keys_input
        )

    except json.JSONDecodeError:

        # ----------------------------------------------------
        # Fallback: comma-separated
        # ----------------------------------------------------

        parsed = [
            item.strip()
            for item in keys_input.split(",")
            if item.strip()
        ]

    # --------------------------------------------------------
    # Nested array support
    # --------------------------------------------------------

    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], list)
    ):

        parsed = parsed[0]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not isinstance(
        parsed,
        list
    ):

        raise HTTPException(
            status_code=400,
            detail="keys must be a JSON array"
        )

    result = []

    for key in parsed:

        if key is None:
            continue

        if not isinstance(
            key,
            str
        ):

            key = str(key)

        key = key.strip()

        if (
            key
            and key not in result
        ):

            result.append(
                key
            )

    return result


# ============================================================
# NORMALIZE KEY
# ============================================================

def normalize_key(
    key: str
) -> str:

    """
    Normalize keys only for internal matching.

    Examples:

    LYMPH%  -> LYMPH%
    LYMPH % -> LYMPH%
    RDW-CV  -> RDWCV
    RDW CV  -> RDWCV
    RDW_CV  -> RDWCV
    """

    value = str(
        key
    ).upper().strip()

    value = value.replace(
        " ",
        ""
    )

    value = value.replace(
        "-",
        ""
    )

    value = value.replace(
        "_",
        ""
    )

    return value


# ============================================================
# BUILD DYNAMIC PROMPT
# ============================================================

def build_prompt(
    requested_keys: list[str],
    column_count: int
) -> str:

    # --------------------------------------------------------
    # Requested keys come ONLY from request
    # --------------------------------------------------------

    requested_keys_json = json.dumps(
        requested_keys,
        ensure_ascii=False
    )

    # --------------------------------------------------------
    # Dynamic JSON example
    # --------------------------------------------------------

    example_result = {
        key: ""
        for key in requested_keys
    }

    example_json = json.dumps(
        example_result,
        ensure_ascii=False
    )

    return f"""
You are a high-accuracy OCR engine for pathology/laboratory
analyzer reports.

Extract ONLY these requested keys:

{requested_keys_json}

columnCount={column_count}

Rules:

- Match each key to its nearby value using physical image
  position.

- Read top-to-bottom and left-to-right.

- Ignore spaces, hyphens and underscores when matching keys.

- LYMPH% may appear as LYMPH %.

- RDW-CV may appear as RDW CV or RDW_CV.

- Ignore H/L/HIGH/LOW/* flags between the key and its value.

- Return ONLY the actual result value.

- Do NOT return units.

- Do NOT return reference ranges.

- Do NOT return H/L flags.

- Do NOT return labels.

- Do NOT return explanations.

- Never guess.

- If a requested key cannot be confidently found,
  return an empty string.

- Use physical image position to associate each parameter
  with its value.

- Do not infer a value from another parameter.

- If multiple columns are present, use columnCount={column_count}
  to understand the layout.

- Return every requested key exactly as provided.

- Do NOT add any key that was not requested.

- Return ONLY valid JSON.

Expected JSON structure:

{example_json}
"""


# ============================================================
# CLEAN GEMINI RESULT
# ============================================================

def clean_result(
    raw_text: str,
    requested_keys: list[str]
) -> dict:

    # --------------------------------------------------------
    # Always return only requested keys
    # --------------------------------------------------------

    result = {
        key: ""
        for key in requested_keys
    }

    if not raw_text:
        return result

    text = raw_text.strip()

    # --------------------------------------------------------
    # Remove markdown fences if necessary
    # --------------------------------------------------------

    if text.startswith("```"):

        if text.startswith("```json"):

            text = text[7:]

        elif text.startswith("```JSON"):

            text = text[7:]

        else:

            text = text[3:]

        if text.endswith("```"):

            text = text[:-3]

        text = text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

    except Exception as e:

        logger.warning(
            "Invalid Gemini JSON: %s",
            e
        )

        return result

    if not isinstance(
        data,
        dict
    ):

        return result

    # --------------------------------------------------------
    # Normalize Gemini response keys
    # --------------------------------------------------------

    normalized_data = {}

    for key, value in data.items():

        normalized_data[
            normalize_key(key)
        ] = value

    # --------------------------------------------------------
    # Extract only requested keys
    # --------------------------------------------------------

    for requested_key in requested_keys:

        normalized_requested = normalize_key(
            requested_key
        )

        value = normalized_data.get(
            normalized_requested,
            ""
        )

        # Do not return complex values
        if isinstance(
            value,
            (dict, list)
        ):

            value = ""

        elif value is None:

            value = ""

        else:

            value = str(
                value
            ).strip()

        result[
            requested_key
        ] = value

    return result


# ============================================================
# TOKEN USAGE LOGGER
# ============================================================

def print_token_usage(
    response
) -> None:

    print()

    print(
        "GEMINI TOKEN USAGE"
    )

    print(
        "=" * 70
    )

    usage = getattr(
        response,
        "usage_metadata",
        None
    )

    if usage is None:

        print(
            "Input tokens: None"
        )

        print(
            "Output tokens: None"
        )

        print(
            "Thinking tokens: None"
        )

        print(
            "Total tokens: None"
        )

    else:

        input_tokens = getattr(
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
            f"Input tokens: {input_tokens}"
        )

        print(
            f"Output tokens: {output_tokens}"
        )

        print(
            f"Thinking tokens: {thinking_tokens}"
        )

        print(
            f"Total tokens: {total_tokens}"
        )

    print(
        "=" * 70
    )

    print()


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

@app.post("/extract")
async def extract_ocr(
    image: UploadFile = File(...),
    keys: str = Form(...),
    columnCount: str = Form(default="1")
):

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        original_bytes = await image.read()

        if not original_bytes:

            raise HTTPException(
                status_code=400,
                detail="Image is empty"
            )

        # ====================================================
        # READ KEYS FROM REQUEST
        # ====================================================

        requested_keys = parse_keys(
            keys
        )

        if not requested_keys:

            raise HTTPException(
                status_code=400,
                detail="keys cannot be empty"
            )

        # ====================================================
        # READ COLUMN COUNT FROM REQUEST
        # ====================================================

        try:

            column_count = int(
                columnCount
            )

        except (
            ValueError,
            TypeError
        ):

            column_count = 1

        if column_count < 1:
            column_count = 1

        # ====================================================
        # LOG REQUEST
        # ====================================================

        logger.info(
            "=" * 70
        )

        logger.info(
            "NEW OCR REQUEST"
        )

        logger.info(
            "Requested keys: %s",
            requested_keys
        )

        logger.info(
            "Column count: %d",
            column_count
        )

        logger.info(
            "Original image size: %.2f KB",
            len(original_bytes) / 1024
        )

        logger.info(
            "=" * 70
        )

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        processed_bytes = preprocess_image(
            original_bytes
        )

        # ====================================================
        # BUILD PROMPT
        # ====================================================

        prompt = build_prompt(
            requested_keys,
            column_count
        )

        # ====================================================
        # GEMINI REQUEST
        # ====================================================

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                types.Part.from_bytes(
                    data=processed_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json"
            )
        )

        # ====================================================
        # PRINT TOKEN USAGE
        # ====================================================

        print_token_usage(
            response
        )

        # ====================================================
        # READ GEMINI RESPONSE
        # ====================================================

        raw_text = response.text

        logger.info(
            "Gemini raw response: %s",
            raw_text
        )

        # ====================================================
        # CLEAN RESULT
        # ====================================================

        result = clean_result(
            raw_text,
            requested_keys
        )

        # ====================================================
        # LOG FINAL RESULT
        # ====================================================

        logger.info(
            "Final OCR result: %s",
            result
        )

        logger.info(
            "=" * 70
        )

        # ====================================================
        # RETURN ONLY REQUESTED KEYS
        # ====================================================

        return result

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "OCR processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8080"
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )

