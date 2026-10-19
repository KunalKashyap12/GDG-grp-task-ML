package com.cropprotect.backend.controller;

import com.cropprotect.backend.dto.CropsCatalogResponse;
import com.cropprotect.backend.dto.FinalPredictionResponse;
import com.cropprotect.backend.dto.GuidanceInfo;
import com.cropprotect.backend.dto.MlPredictionResponse;
import com.cropprotect.backend.exception.InvalidImageException;
import com.cropprotect.backend.service.GuidanceService;
import com.cropprotect.backend.service.MlClientService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api")
public class PredictController {

    private static final Logger log = LoggerFactory.getLogger(PredictController.class);
    private static final long MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
    private static final List<String> ALLOWED_CONTENT_TYPES = Arrays.asList(
            "image/jpeg",
            "image/jpg",
            "image/png"
    );

    private final MlClientService mlClientService;
    private final GuidanceService guidanceService;

    public PredictController(MlClientService mlClientService, GuidanceService guidanceService) {
        this.mlClientService = mlClientService;
        this.guidanceService = guidanceService;
    }

    /**
     * POST /api/predict
     * Accepts a leaf photo and optional crop hint, runs ML inference, and attaches
     * actionable disease preventive guidance.
     */
    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<FinalPredictionResponse> predict(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "crop", required = false) String cropHint
    ) {
        // 1. Validation
        if (image == null || image.isEmpty()) {
            throw new InvalidImageException("Uploaded file is empty. Please provide a clear leaf image.");
        }

        if (image.getSize() > MAX_FILE_SIZE) {
            throw new InvalidImageException(String.format(
                    "Image file size (%.2f MB) exceeds allowed limit of 5 MB.",
                    image.getSize() / (1024.0 * 1024.0)
            ));
        }

        String contentType = image.getContentType();
        String rawFilename = image.getOriginalFilename();
        String originalFilename = (rawFilename != null) ? rawFilename.toLowerCase() : "";

        boolean isAllowedType = (contentType != null && ALLOWED_CONTENT_TYPES.contains(contentType.toLowerCase())) ||
                originalFilename.endsWith(".jpg") || originalFilename.endsWith(".jpeg") || originalFilename.endsWith(".png");

        if (!isAllowedType) {
            throw new InvalidImageException(
                    "Invalid file type '" + contentType + "'. Supported image formats are JPEG and PNG."
            );
        }

        log.info("Processing prediction request: filename={}, size={} bytes, cropHint={}",
                rawFilename, image.getSize(), cropHint);

        // 2. ML Service Call
        MlPredictionResponse mlResponse = mlClientService.predict(image, cropHint);

        // 3. Attach preventive guidance if prediction resolved to a valid class
        GuidanceInfo guidance = null;
        if ("ok".equalsIgnoreCase(mlResponse.getStatus()) || "low_confidence".equalsIgnoreCase(mlResponse.getStatus())) {
            Optional<GuidanceInfo> guidanceOpt = guidanceService.getGuidance(
                    mlResponse.getCrop(),
                    mlResponse.getDisease()
            );
            guidance = guidanceOpt.orElse(null);
        }

        // 4. Construct Final Decision Support Response
        FinalPredictionResponse response = FinalPredictionResponse.from(mlResponse, guidance);
        return ResponseEntity.ok(response);
    }

    /**
     * GET /api/crops
     * Proxies the ML service classes registry to support dynamic crop and disease dropdowns in frontends.
     */
    @GetMapping("/crops")
    public ResponseEntity<CropsCatalogResponse> getSupportedCrops() {
        CropsCatalogResponse catalog = mlClientService.getCropsCatalog();
        return ResponseEntity.ok(catalog);
    }
}
