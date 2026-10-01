package com.OCR.Reader.constants;

public class AppConstants {

    public static final String SUCCESS = "SUCCESS";

    public static final String ERROR = "ERROR";

    // OCR languages: English + Hindi
    public static final String LANGUAGE = "eng+hin";

    // Tesseract tessdata path
    // Can be overridden using TESSDATA_PATH environment variable
    public static final String PATH = System.getenv().getOrDefault(
            "TESSDATA_PATH",
            "/usr/share/tesseract-ocr/5/tessdata"
    );
}