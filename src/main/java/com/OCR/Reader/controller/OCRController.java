package com.OCR.Reader.controller;

import com.OCR.Reader.pojo.OCRResult;
import com.OCR.Reader.service.OCRService;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/ocr")
public class OCRController {

	@Autowired
	private OCRService ocrService;

	// Endpoint for image upload and OCR processing
	@PostMapping(value = "/extract", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public OCRResult processImage(@RequestParam("image") MultipartFile file, @RequestParam("keys") List<String> keys) {

		return ocrService.processImage(file, keys);
	}

	@GetMapping("/")
	public String sayHello() {
		return "User Service Working Amol!";
	}
}
