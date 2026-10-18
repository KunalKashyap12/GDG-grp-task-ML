package com.cropprotect.backend.service;

import com.cropprotect.backend.dto.CropsCatalogResponse;
import com.cropprotect.backend.dto.MlPredictionResponse;
import com.cropprotect.backend.exception.MlServiceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.util.retry.Retry;

import java.io.IOException;
import java.time.Duration;

@Service
public class MlClientService {

    private static final Logger log = LoggerFactory.getLogger(MlClientService.class);
    private final WebClient mlWebClient;

    public MlClientService(WebClient mlWebClient) {
        this.mlWebClient = mlWebClient;
    }

    /**
     * Submits image and optional crop hint to FastAPI ML service (/predict).
     * Includes timeout handling, 1 automatic retry on network drops, and error translation.
     */
    public MlPredictionResponse predict(MultipartFile file, String cropHint) {
        log.debug("Submitting prediction request to ML service. File: {}, CropHint: {}", file.getOriginalFilename(), cropHint);

        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        try {
            ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename() != null ? file.getOriginalFilename() : "leaf_photo.jpg";
                }
            };
            bodyBuilder.part("image", resource).header("Content-Type", file.getContentType());
        } catch (IOException e) {
            throw new MlServiceException("Failed to read image bytes: " + e.getMessage(), e);
        }

        if (cropHint != null && !cropHint.trim().isEmpty()) {
            bodyBuilder.part("crop", cropHint.trim());
        }

        try {
            return mlWebClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                    .retrieve()
                    .bodyToMono(MlPredictionResponse.class)
                    .retryWhen(Retry.fixedDelay(1, Duration.ofMillis(500))
                            .filter(throwable -> throwable instanceof WebClientRequestException))
                    .block(Duration.ofSeconds(10));
        } catch (WebClientResponseException ex) {
            log.error("ML service returned HTTP error {}: {}", ex.getStatusCode(), ex.getResponseBodyAsString());
            throw new MlServiceException("ML service error: " + ex.getResponseBodyAsString(), ex.getStatusCode().value());
        } catch (WebClientRequestException ex) {
            log.error("Failed to connect to ML service: {}", ex.getMessage());
            throw new MlServiceException("ML service is unreachable. Ensure ml_service container is running.", ex);
        } catch (Exception ex) {
            log.error("ML service invocation timed out or encountered an unexpected error: {}", ex.getMessage());
            throw new MlServiceException("ML service failed to respond within time limit.", 504);
        }
    }

    /**
     * Proxies catalog of supported crops and disease classes from FastAPI /classes.
     */
    public CropsCatalogResponse getCropsCatalog() {
        log.debug("Fetching supported crops and disease classes from ML service");
        try {
            return mlWebClient.get()
                    .uri("/classes")
                    .retrieve()
                    .bodyToMono(CropsCatalogResponse.class)
                    .retryWhen(Retry.fixedDelay(1, Duration.ofMillis(300))
                            .filter(t -> t instanceof WebClientRequestException))
                    .block(Duration.ofSeconds(5));
        } catch (Exception ex) {
            log.error("Failed to fetch classes catalog from ML service: {}", ex.getMessage());
            throw new MlServiceException("Could not retrieve supported crops catalog from ML service.", ex);
        }
    }
}
