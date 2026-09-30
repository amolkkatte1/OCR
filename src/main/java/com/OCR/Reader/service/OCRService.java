package com.OCR.Reader.service;

import com.OCR.Reader.pojo.OCRResult;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface OCRService {

	OCRResult processImage(MultipartFile file, List<String> keys);

}