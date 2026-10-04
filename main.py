import os
import json
import re
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.8-flash"


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gemini OCR API",
    version="1.0.0"
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
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_json_response(text: str) -> Dict[str, Any]:
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
    ```
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

    # Try direct JSON
    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
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

            parsed = json.loads(json_text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON returned by Gemini: {str(e)}"
            )

    raise ValueError(
        "Gemini response does not contain a valid JSON object"
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

    prompt = f"""
You are an OCR extraction engine for pathology and laboratory analyzer reports.

You will receive an image containing laboratory test results.

Your task is to extract ONLY the requested keys.

REQUESTED KEYS:
{keys_json}

COLUMN COUNT:
{column_count}

IMPORTANT LAYOUT RULE:

The image can contain multiple key-value columns on the same physical row.

columnCount = {column_count}

means that each physical row can contain up to {column_count} key-value columns.

For example, if columnCount = 2:

WBC     5.2        HGB     10.7

This is ONE physical row containing TWO key-value columns.

Therefore:

WBC -> 5.2
HGB -> 10.7

Do NOT associate a value from another column with the wrong key.

Read the image:

1. From top to bottom.
2. Within each row, from left to right.
3. Treat each key and its nearby value as one key-value pair.
4. Respect the physical position of the text in the image.
5. Do not reorder values based only on OCR text order.
6. Use the nearest corresponding value belonging to each key.

KEY MATCHING RULES:

Keys may appear with small formatting differences.

Examples:

LYMPH% and LYMPH % are the same key.

RDW-CV and RDW CV are the same key.

RDW-CV and RDW-CV% should NOT automatically be treated as the same
unless the image clearly shows that they represent the requested field.

Ignore spaces when matching keys.

Ignore hyphens and underscores when matching keys.

The report may contain H or L flags between the key and value.

For example:

WBC 5.2
WBC H 5.2
WBC L 5.2

In all cases:

WBC -> 5.2

The H/L flag is NOT part of the value.

Do not include:

H
L
HIGH
LOW
*
flags
reference ranges
units

unless they are actually part of the requested value.

VALUE RULES:

Return the actual result value associated with the requested key.

Examples:

WBC 5.2 -> "5.2"

HGB 10.7 -> "10.7"

RBC 4.03 -> "4.03"

PLT 171 -> "171"

Do not return reference ranges.

Example:

WBC 5.2 4.0-10.0

Return:

"WBC": "5.2"

NOT:

"WBC": "5.2 4.0-10.0"

If a requested key cannot be found confidently, return an empty string.

Do not guess.

OUTPUT REQUIREMENT:

Return ONLY a valid JSON object.

Do NOT return:

- markdown
- ```json
- explanations
- comments
- arrays
- additional fields

The JSON object must contain ONLY the requested keys.

Example:

{{
    "WBC": "5.2",
    "HGB": "10.7",
    "RBC": "4.03"
}}

If a value cannot be found:

{{
    "WBC": "5.2",
    "HGB": "",
    "RBC": "4.03"
}}

VERY IMPORTANT:

Return every requested key exactly as provided in REQUESTED KEYS.

Do not rename the keys.

Do not add keys that were not requested.
"""

    return prompt


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

    print()
    print("=" * 70)
    print("NEW OCR REQUEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate columnCount
    # --------------------------------------------------------

    if columnCount <= 0:

        print("ERROR: columnCount must be greater than 0")

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # --------------------------------------------------------
    # Parse keys
    # --------------------------------------------------------

    try:

        requested_keys = parse_keys(keys)

    except Exception as e:

        print("=" * 70)
        print("KEY PARSING ERROR")
        print(type(e).__name__)
        print(str(e))
        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    print("Requested keys:")
    print(requested_keys)

    print("Column count:")
    print(columnCount)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await image.read()

    except Exception as e:

        print("=" * 70)
        print("IMAGE READ ERROR")
        print(type(e).__name__)
        print(str(e))
        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    if not image_bytes:

        print("ERROR: Empty image")

        return OCRResponse(
            status="ERROR",
            data={}
        )

    print("Image filename:")
    print(image.filename)

    print("Image content type:")
    print(image.content_type)

    print("Image size:")
    print(len(image_bytes), "bytes")

    # --------------------------------------------------------
    # Validate MIME type
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = create_prompt(
        requested_keys,
        columnCount
    )

    print()
    print("Gemini prompt created.")
    print("Model:", MODEL_NAME)

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    try:

        print()
        print("=" * 70)
        print("CALLING GEMINI")
        print("=" * 70)

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[

                types.Part.from_text(
                    text=prompt
                ),

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type
                )
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json"
            )
        )

        print()
        print("=" * 70)
        print("GEMINI RESPONSE RECEIVED")
        print("=" * 70)

        response_text = response.text

        print(response_text)

        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("GEMINI ERROR")
        print("=" * 70)

        print("Error type:")
        print(type(e).__name__)

        print("Error:")
        print(str(e))

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # --------------------------------------------------------
    # Parse Gemini JSON
    # --------------------------------------------------------

    try:

        print()
        print("=" * 70)
        print("PARSING GEMINI JSON")
        print("=" * 70)

        gemini_data = clean_json_response(
            response_text
        )

        print("Parsed Gemini data:")
        print(
            json.dumps(
                gemini_data,
                indent=4,
                ensure_ascii=False
            )
        )

        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("JSON PARSING ERROR")
        print("=" * 70)

        print("Error type:")
        print(type(e).__name__)

        print("Error:")
        print(str(e))

        print("Gemini raw response:")
        print(response_text)

        print("=" * 70)

        return OCRResponse(
            status="ERROR",
            data={}
        )

    # --------------------------------------------------------
    # Build final response
    # --------------------------------------------------------

    final_data = {}

    for requested_key in requested_keys:

        value = find_value_for_key(
            gemini_data,
            requested_key
        )

        final_data[requested_key] = value

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

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
    print()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return OCRResponse(
        status="SUCCESS",
        data=final_data
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )