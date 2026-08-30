# ============================================================
# IMPORTANT:
# Disable Paddle oneDNN/MKLDNN BEFORE importing paddleocr
# ============================================================

import os

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"


# ============================================================
# IMPORTS
# ============================================================

import re
import uuid
import shutil
import json

from typing import Dict, Any, List

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form
)

from fastapi.responses import JSONResponse

from paddleocr import PaddleOCR


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Image Key Value OCR API",
    description="Extract requested key-value data from image",
    version="3.0.0"
)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# OCR INITIALIZATION
# ============================================================

print("==============================================")
print("Loading PaddleOCR model...")
print("==============================================")

try:

    ocr = PaddleOCR(
        lang="en",

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False
    )

    print("PaddleOCR loaded successfully.")

except Exception as e:

    print("==============================================")
    print("PADDLE OCR INITIALIZATION ERROR")
    print("==============================================")

    print(str(e))

    print("==============================================")

    raise


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "Image OCR API is running",
        "endpoint": "POST /extract"
    }


@app.get("/health")
def health():

    return {
        "status": "UP"
    }


# ============================================================
# CLEAN KEY
# ============================================================

def clean_key(
        key: str
) -> str:

    if not key:
        return ""

    key = key.strip()

    key = re.sub(
        r"^[\-\:\=\.\,\s]+",
        "",
        key
    )

    key = re.sub(
        r"[\:\=\-\.\,\s]+$",
        "",
        key
    )

    return key.strip()


# ============================================================
# CLEAN VALUE
# ============================================================

def clean_value(
        value: str
) -> str:

    if not value:
        return ""

    value = value.strip()

    value = re.sub(
        r"^[\:\=\-\.\,\s]+",
        "",
        value
    )

    return value.strip()


# ============================================================
# CLEAN NUMERIC VALUE
# ============================================================

def clean_numeric_value(
        value: str
) -> str:

    if not value:
        return ""

    value = value.strip()

    value = re.sub(
        r"^[\s:=\-]+",
        "",
        value
    )

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        value
    )

    if match:

        return match.group(0)

    return ""


# ============================================================
# CHECK NUMERIC VALUE
# ============================================================

def is_numeric_value(
        text: str
) -> bool:

    if not text:
        return False

    return bool(
        re.search(
            r"\d",
            text
        )
    )


# ============================================================
# NORMALIZE KEY
# ============================================================

def normalize_key(
        text: str
) -> str:

    if not text:
        return ""

    text = text.strip().upper()

    # Remove spaces
    text = re.sub(
        r"\s+",
        "",
        text
    )

    # ========================================================
    # OCR CORRECTIONS
    # ========================================================

    corrections = {

        "WBC": "WBC",

        "LYMPH": "LYMPH#",
        "LYMPH#": "LYMPH#",

        "MID": "MID#",
        "MID#": "MID#",

        "GRAN": "GRAN#",
        "GRAN#": "GRAN#",

        "LYMPH%": "LYMPH%",
        "MID%": "MID%",
        "GRAN%": "GRAN%",

        "HGB": "HGB",

        "RBC": "RBC",
        "RB": "RBC",

        "HCT": "HCT",

        "MCV": "MCV",

        "MCH": "MCH",

        "MCHC": "MCHC",

        "RDW-CV": "RDW-CV",
        "RDWCV": "RDW-CV",

        "RDW-SD": "RDW-SD",
        "RDWSD": "RDW-SD",

        "PLT": "PLT",

        "MPV": "MPV",

        "PDW": "PDW",

        "PCT": "PCT"
    }

    return corrections.get(
        text,
        text
    )


# ============================================================
# NORMALIZE REQUESTED KEYS
# ============================================================

def normalize_requested_keys(
        keys: List[str]
) -> List[str]:

    normalized = []

    for key in keys:

        if not key:
            continue

        key = key.strip()

        normalized_key = normalize_key(
            key
        )

        # ----------------------------------------------------
        # Preserve ID / Time / Next ID exactly
        # ----------------------------------------------------

        if normalized_key in {
            "ID",
            "TIME",
            "NEXTID"
        }:

            if normalized_key == "NEXTID":
                normalized_key = "Next ID"

            elif normalized_key == "TIME":
                normalized_key = "Time"

            elif normalized_key == "ID":
                normalized_key = "ID"

        if normalized_key not in normalized:

            normalized.append(
                normalized_key
            )

    return normalized


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

def extract_text_from_image(
        image_path: str
) -> List[str]:

    all_text = []

    print("Running OCR...")

    try:

        result = ocr.predict(
            image_path
        )

        print("OCR completed.")

        # ====================================================
        # PROCESS OCR RESULT
        # ====================================================

        for res in result:

            try:

                data = None

                # ------------------------------------------------
                # res.json
                # ------------------------------------------------

                try:

                    data = res.json

                except Exception:

                    data = None

                # ------------------------------------------------
                # If callable
                # ------------------------------------------------

                if callable(data):

                    data = data()

                # ------------------------------------------------
                # JSON string
                # ------------------------------------------------

                if isinstance(
                        data,
                        str
                ):

                    try:

                        data = json.loads(
                            data
                        )

                    except Exception:

                        data = None

                # ------------------------------------------------
                # Dictionary
                # ------------------------------------------------

                if isinstance(
                        data,
                        dict
                ):

                    result_data = data.get(
                        "res",
                        data
                    )

                    texts = result_data.get(
                        "rec_texts",
                        []
                    )

                    for text in texts:

                        if text is None:
                            continue

                        text = str(
                            text
                        ).strip()

                        if text:

                            all_text.append(
                                text
                            )

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                if not data:

                    try:

                        data = res["res"]

                        texts = data.get(
                            "rec_texts",
                            []
                        )

                        for text in texts:

                            if text is None:
                                continue

                            text = str(
                                text
                            ).strip()

                            if text:

                                all_text.append(
                                    text
                                )

                    except Exception:

                        pass

            except Exception as e:

                print(
                    "OCR result parsing error:",
                    str(e)
                )

    except Exception as e:

        print(
            "OCR execution error:",
            str(e)
        )

        raise

    # ========================================================
    # CLEAN OCR LINES
    # ========================================================

    cleaned_text = []

    for text in all_text:

        text = text.strip()

        if not text:
            continue

        cleaned_text.append(
            text
        )

    # ========================================================
    # PRINT OCR RESULT
    # ========================================================

    print("==============================================")
    print("Detected OCR lines")
    print("==============================================")

    for line in cleaned_text:

        print(
            " ->",
            line
        )

    print("==============================================")

    return cleaned_text


# ============================================================
# CHECK IF LINE IS REQUESTED KEY
# ============================================================

def find_requested_key(
        line: str,
        requested_keys: List[str]
):

    if not line:
        return None

    normalized_line = normalize_key(
        line
    )

    # ========================================================
    # DIRECT MATCH
    # ========================================================

    for requested_key in requested_keys:

        if normalized_line == requested_key:

            return requested_key

    return None


# ============================================================
# EXTRACT VALUE FROM SAME LINE
# ============================================================

def extract_value_from_same_line(
        line: str,
        requested_key: str
) -> str:

    if not line:
        return ""

    # ========================================================
    # ID:15
    # ========================================================

    if ":" in line:

        parts = line.split(
            ":",
            1
        )

        key_part = normalize_key(
            parts[0]
        )

        if (
            key_part == requested_key
            or
            (
                requested_key == "Next ID"
                and key_part == "NEXTID"
            )
        ):

            return clean_value(
                parts[1]
            )

    # ========================================================
    # ID=15
    # ========================================================

    if "=" in line:

        parts = line.split(
            "=",
            1
        )

        key_part = normalize_key(
            parts[0]
        )

        if key_part == requested_key:

            return clean_value(
                parts[1]
            )

    # ========================================================
    # WBC 8.8
    # ========================================================

    escaped_key = re.escape(
        requested_key
    )

    pattern = (
        r"^\s*"
        + escaped_key
        + r"\s+(.+)$"
    )

    match = re.match(
        pattern,
        line,
        re.IGNORECASE
    )

    if match:

        return clean_value(
            match.group(1)
        )

    return ""


# ============================================================
# EXTRACT NORMAL KEY VALUE
# ============================================================

def extract_normal_requested_values(
        lines: List[str],
        requested_keys: List[str]
) -> Dict[str, Any]:

    result = {}

    for i, line in enumerate(lines):

        line = line.strip()

        if not line:
            continue

        # ====================================================
        # CHECK EVERY REQUESTED KEY
        # ====================================================

        for requested_key in requested_keys:

            # ------------------------------------------------
            # Same-line value
            # ------------------------------------------------

            value = extract_value_from_same_line(
                line,
                requested_key
            )

            if value:

                result[requested_key] = value

                continue

            # ------------------------------------------------
            # Key on separate line
            # ------------------------------------------------

            found_key = find_requested_key(
                line,
                requested_keys
            )

            if found_key != requested_key:

                continue

            # ------------------------------------------------
            # Search next few lines
            # ------------------------------------------------

            for j in range(
                    i + 1,
                    min(
                        i + 5,
                        len(lines)
                    )
            ):

                next_line = lines[j].strip()

                if not next_line:
                    continue

                # --------------------------------------------
                # Stop if next line is another requested key
                # --------------------------------------------

                another_key = find_requested_key(
                    next_line,
                    requested_keys
                )

                if another_key:

                    break

                # --------------------------------------------
                # ID / Time / Next ID
                # --------------------------------------------

                if requested_key in {
                    "ID",
                    "Time",
                    "Next ID"
                }:

                    # Use any non-empty text
                    # except obvious noise

                    if next_line:

                        result[requested_key] = (
                            clean_value(
                                next_line
                            )
                        )

                        break

                # --------------------------------------------
                # Numeric parameter
                # --------------------------------------------

                else:

                    if is_numeric_value(
                            next_line
                    ):

                        numeric_value = (
                            clean_numeric_value(
                                next_line
                            )
                        )

                        if numeric_value:

                            result[requested_key] = (
                                numeric_value
                            )

                            break

    return result


# ============================================================
# EXTRACT MEDICAL VALUES
# ============================================================

def extract_medical_requested_values(
        lines: List[str],
        requested_keys: List[str]
) -> Dict[str, Any]:

    result = {}

    # ========================================================
    # MEDICAL KEYS
    # ========================================================

    medical_keys = {

        "WBC",

        "LYMPH#",
        "MID#",
        "GRAN#",

        "LYMPH%",
        "MID%",
        "GRAN%",

        "HGB",
        "RBC",
        "HCT",

        "MCV",
        "MCH",
        "MCHC",

        "RDW-CV",
        "RDW-SD",

        "PLT",
        "MPV",
        "PDW",
        "PCT"
    }

    # ========================================================
    # ONLY REQUESTED MEDICAL KEYS
    # ========================================================

    medical_keys = (
        medical_keys
        .intersection(
            set(requested_keys)
        )
    )

    # ========================================================
    # LOOP OCR LINES
    # ========================================================

    for i in range(
            len(lines)
    ):

        current_line = lines[i].strip()

        if not current_line:
            continue

        current_key = normalize_key(
            current_line
        )

        # ====================================================
        # NOT MEDICAL KEY
        # ====================================================

        if current_key not in medical_keys:

            continue

        # ====================================================
        # SEARCH NEXT LINES
        # ====================================================

        for j in range(
                i + 1,
                min(
                    i + 5,
                    len(lines)
                )
        ):

            next_line = lines[j].strip()

            if not next_line:
                continue

            # =================================================
            # ANOTHER REQUESTED KEY
            # =================================================

            next_key = normalize_key(
                next_line
            )

            if next_key in requested_keys:

                break

            # =================================================
            # IGNORE RANDOM OCR LETTERS
            #
            # Example:
            #
            # WBC
            # 8.8×10/L
            #
            # MID#
            # W
            # 0.5×10/L
            # =================================================

            if not is_numeric_value(
                    next_line
            ):

                continue

            # =================================================
            # GET NUMERIC VALUE
            # =================================================

            numeric_value = clean_numeric_value(
                next_line
            )

            if not numeric_value:

                continue

            # =================================================
            # DON'T OVERWRITE
            # =================================================

            if current_key not in result:

                result[current_key] = (
                    numeric_value
                )

            break

    return result


# ============================================================
# FINAL REQUESTED KEY VALUE EXTRACTION
# ============================================================

def extract_requested_key_values(
        lines: List[str],
        requested_keys: List[str]
) -> Dict[str, Any]:

    result = {}

    # ========================================================
    # NORMAL VALUES
    # ========================================================

    normal_values = extract_normal_requested_values(
        lines,
        requested_keys
    )

    result.update(
        normal_values
    )

    # ========================================================
    # MEDICAL VALUES
    # ========================================================

    medical_values = extract_medical_requested_values(
        lines,
        requested_keys
    )

    result.update(
        medical_values
    )

    # ========================================================
    # PRESERVE REQUESTED KEY ORDER
    # ========================================================

    ordered_result = {}

    for key in requested_keys:

        if key in result:

            ordered_result[key] = result[key]

    return ordered_result


# ============================================================
# PARSE KEYS FROM POSTMAN
# ============================================================

def parse_keys(
        keys_string: str
) -> List[str]:

    if not keys_string:

        raise HTTPException(
            status_code=400,
            detail="keys is required"
        )

    keys_string = keys_string.strip()

    # ========================================================
    # TRY JSON ARRAY
    #
    # [
    #   "ID",
    #   "Time",
    #   "WBC"
    # ]
    # ========================================================

    try:

        parsed = json.loads(
            keys_string
        )

        if isinstance(
                parsed,
                list
        ):

            keys = []

            for item in parsed:

                if isinstance(
                        item,
                        str
                ):

                    item = item.strip()

                    if item:

                        keys.append(
                            item
                        )

            if not keys:

                raise HTTPException(
                    status_code=400,
                    detail="keys array is empty"
                )

            return normalize_requested_keys(
                keys
            )

    except json.JSONDecodeError:

        pass

    # ========================================================
    # FALLBACK
    #
    # ID,Time,Next ID,WBC,HGB
    # ========================================================

    keys = [
        item.strip()
        for item in keys_string.split(",")
        if item.strip()
    ]

    if not keys:

        raise HTTPException(
            status_code=400,
            detail="No valid keys found"
        )

    return normalize_requested_keys(
        keys
    )


# ============================================================
# OCR API
# ============================================================

@app.post("/extract")
async def extract(

        image: UploadFile = File(...),

        keys: str = Form(...)
):

    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    if not image.filename:

        raise HTTPException(
            status_code=400,
            detail="Image file is required"
        )

    # ========================================================
    # PARSE REQUESTED KEYS
    # ========================================================

    requested_keys = parse_keys(
        keys
    )

    print("==============================================")
    print("REQUESTED KEYS")
    print("==============================================")

    print(
        json.dumps(
            requested_keys,
            indent=4
        )
    )

    # ========================================================
    # ALLOWED EXTENSIONS
    # ========================================================

    allowed_extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    extension = os.path.splitext(
        image.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG, PNG, BMP "
                "and WEBP images are supported"
            )
        )

    # ========================================================
    # UNIQUE FILE NAME
    # ========================================================

    filename = (
        str(uuid.uuid4())
        + extension
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    print("==============================================")
    print("New OCR Request")
    print("Original File:", image.filename)
    print("Temporary File:", image_path)
    print("==============================================")

    try:

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        with open(
                image_path,
                "wb"
        ) as buffer:

            shutil.copyfileobj(
                image.file,
                buffer
            )

        print(
            "Image saved successfully."
        )

        # ====================================================
        # OCR
        # ====================================================

        lines = extract_text_from_image(
            image_path
        )

        # ====================================================
        # EXTRACT ONLY REQUESTED KEYS
        # ====================================================

        key_values = extract_requested_key_values(
            lines,
            requested_keys
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success": True,

            "filename": image.filename,

            "requested_keys": requested_keys,

            "data": key_values,

            "raw_text": lines
        }

        # ====================================================
        # PRINT RESPONSE
        # ====================================================

        print("==============================================")
        print("FINAL OCR RESPONSE")
        print("==============================================")

        print(
            json.dumps(
                response,
                indent=4,
                ensure_ascii=False
            )
        )

        print("==============================================")

        return JSONResponse(
            status_code=200,
            content=response
        )

    except HTTPException:

        raise

    except Exception as e:

        print("==============================================")
        print("OCR ERROR")
        print("==============================================")

        print(
            str(e)
        )

        print("==============================================")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ====================================================
        # DELETE TEMP IMAGE
        # ====================================================

        if os.path.exists(
                image_path
        ):

            try:

                os.remove(
                    image_path
                )

                print(
                    "Temporary image deleted."
                )

            except Exception as e:

                print(
                    "Unable to delete temporary image:",
                    str(e)
                )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )