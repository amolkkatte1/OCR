package com.OCR.Reader.service.impl;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.OCR.Reader.constants.AppConstants;
import com.OCR.Reader.pojo.OCRResult;
import com.OCR.Reader.service.OCRService;
import com.OCR.Reader.util.OCRProcessor;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class OCRServiceImpl implements OCRService {
	@Override
	public OCRResult processImage(MultipartFile file, List<String> keys) {

		OCRResult result = new OCRResult();

		File tempFile = null;

		try {

			// Validate file
			if (file == null || file.isEmpty()) {
				result.setStatus(AppConstants.ERROR);
				return result;
			}

			// Validate keys
			if (keys == null || keys.isEmpty()) {
				result.setStatus(AppConstants.ERROR);
				return result;
			}

			// Create temporary file
			tempFile = File.createTempFile("ocr", ".tmp");

			// Save uploaded image
			file.transferTo(tempFile);

			// Perform OCR
			OCRProcessor ocrProcessor = new OCRProcessor();

			String extractedText = ocrProcessor.extractTextFromImage(tempFile);

			// Extract requested key-value pairs
			Map<String, String> keyValues = extractKeyValues(extractedText, keys);

			// Set response
			result.setData(keyValues);
			result.setStatus(AppConstants.SUCCESS);

		} catch (IOException e) {

			log.error("Error in processing image", e);

			result.setStatus(AppConstants.ERROR);

		} catch (Exception e) {

			log.error("Unexpected error in OCR processing", e);

			result.setStatus(AppConstants.ERROR);

		} finally {

			// Delete temporary file
			if (tempFile != null && tempFile.exists()) {
				if (!tempFile.delete()) {
					log.warn("Unable to delete temporary file: {}", tempFile.getAbsolutePath());
				}
			}
		}

		return result;
	}

	private Map<String, String> extractKeyValues(String extractedText, List<String> keys) {

		Map<String, String> result = new LinkedHashMap<>();

		if (extractedText == null || extractedText.isBlank() || keys == null || keys.isEmpty()) {
			return result;
		}

		String[] lines = extractedText.split("\\r?\\n");

		for (String key : keys) {

			if (key == null || key.isBlank()) {
				continue;
			}

			String searchKey = key.trim();
			searchKey = key.trim().replace("\"", "");
			searchKey = searchKey.contains("[") ? searchKey.substring(1, searchKey.length()) : searchKey;
			for (String line : lines) {

				line = line.trim();

				if (line.isEmpty()) {
					continue;
				}

				/*
				 * Match key only at the beginning of the line.
				 *
				 * Examples:
				 *
				 * WBC 6.2 * 109 RBC 3.13 x 191% PCT 0.143 %
				 */

				if (!(line.toLowerCase().startsWith(searchKey.toLowerCase()))) {
					continue;
				}

				// Remove the key from the beginning
				String remaining = line.substring(searchKey.length()).trim();

				/*
				 * Extract the first numeric value.
				 *
				 * Examples:
				 *
				 * 6.2 * 109 -> 6.2 3.13 x 191% -> 3.13 0.143 % -> 0.143 28.8 % -> 28.8
				 */

				java.util.regex.Matcher matcher = java.util.regex.Pattern.compile("[-+]?\\d+(?:\\.\\d+)?")
						.matcher(remaining);

				if (matcher.find()) {

					String value = matcher.group();

					result.put(searchKey, value);

					break;
				}
			}
		}

		return result;
	}

}
