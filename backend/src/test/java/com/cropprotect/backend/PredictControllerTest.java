package com.cropprotect.backend;

import com.cropprotect.backend.dto.*;
import com.cropprotect.backend.exception.MlServiceException;
import com.cropprotect.backend.service.MlClientService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@SuppressWarnings("null")
class PredictControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private MlClientService mlClientService;

    @Test
    @DisplayName("POST /api/predict returns 200 with prediction, guidance, and disclaimer on valid image")
    void testSuccessfulPredictionWithGuidance() throws Exception {
        // Arrange Mock ML Service response
        MlPredictionResponse mockMlResponse = new MlPredictionResponse();
        mockMlResponse.setStatus("ok");
        mockMlResponse.setCrop("Tomato");
        mockMlResponse.setDisease("Early_blight");
        mockMlResponse.setConfidence(0.94);
        mockMlResponse.setCertainty("high");
        mockMlResponse.setMessage("High confidence prediction.");
        mockMlResponse.setQuality(new QualityInfo(185.4, 128.0));
        mockMlResponse.setTop3(List.of(
                new TopPrediction("Tomato___Early_blight", 0.94),
                new TopPrediction("Tomato___Late_blight", 0.04),
                new TopPrediction("Tomato___Septoria_leaf_spot", 0.01)
        ));

        when(mlClientService.predict(any(), eq("Tomato"))).thenReturn(mockMlResponse);

        MockMultipartFile imageFile = new MockMultipartFile(
                "image",
                "tomato_leaf.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                "fake-jpeg-image-bytes".getBytes()
        );

        // Act & Assert
        mockMvc.perform(multipart("/api/predict")
                        .file(imageFile)
                        .param("crop", "Tomato"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status", is("ok")))
                .andExpect(jsonPath("$.crop", is("Tomato")))
                .andExpect(jsonPath("$.disease", is("Early_blight")))
                .andExpect(jsonPath("$.confidence", is(0.94)))
                .andExpect(jsonPath("$.certainty", is("high")))
                .andExpect(jsonPath("$.guidance.disease_name", containsString("Early Blight")))
                .andExpect(jsonPath("$.guidance.preventive_measures", hasSize(greaterThan(0))))
                .andExpect(jsonPath("$.disclaimer", containsString("automated recommendation")))
                .andExpect(jsonPath("$.timestamp", notNullValue()));
    }

    @Test
    @DisplayName("POST /api/predict returns poor_image_quality without guidance on blurry images")
    void testPoorQualityImagePrediction() throws Exception {
        MlPredictionResponse mockMlResponse = new MlPredictionResponse();
        mockMlResponse.setStatus("poor_image_quality");
        mockMlResponse.setCrop("unknown");
        mockMlResponse.setDisease("unknown");
        mockMlResponse.setConfidence(0.0);
        mockMlResponse.setCertainty("low");
        mockMlResponse.setMessage("Image is blurry (sharpness score 45.2 < 100.0).");
        mockMlResponse.setQuality(new QualityInfo(45.2, 110.0));

        when(mlClientService.predict(any(), any())).thenReturn(mockMlResponse);

        MockMultipartFile imageFile = new MockMultipartFile(
                "image",
                "blurry_leaf.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                "blurry-bytes".getBytes()
        );

        mockMvc.perform(multipart("/api/predict").file(imageFile))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status", is("poor_image_quality")))
                .andExpect(jsonPath("$.message", containsString("blurry")))
                .andExpect(jsonPath("$.guidance").doesNotExist());
    }

    @Test
    @DisplayName("POST /api/predict rejects invalid file type with 400 Bad Request")
    void testInvalidFileType() throws Exception {
        MockMultipartFile pdfFile = new MockMultipartFile(
                "image",
                "document.pdf",
                MediaType.APPLICATION_PDF_VALUE,
                "pdf-data".getBytes()
        );

        mockMvc.perform(multipart("/api/predict").file(pdfFile))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("INVALID_IMAGE_PAYLOAD")))
                .andExpect(jsonPath("$.status", is(400)));
    }

    @Test
    @DisplayName("POST /api/predict translates ML service downtime into clean 503 JSON")
    void testMlServiceDowntimeErrorHandling() throws Exception {
        when(mlClientService.predict(any(), any()))
                .thenThrow(new MlServiceException("ML service is unreachable.", 503));

        MockMultipartFile imageFile = new MockMultipartFile(
                "image",
                "leaf.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                "bytes".getBytes()
        );

        mockMvc.perform(multipart("/api/predict").file(imageFile))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error", is("ML_SERVICE_UNAVAILABLE")))
                .andExpect(jsonPath("$.message", containsString("unreachable")))
                .andExpect(jsonPath("$.status", is(503)));
    }

    @Test
    @DisplayName("GET /api/crops proxies classes catalog")
    void testGetCropsCatalog() throws Exception {
        CropsCatalogResponse mockCatalog = new CropsCatalogResponse(
                Map.of("Tomato", List.of("Early_blight", "Late_blight", "Healthy")),
                3,
                List.of("Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Healthy")
        );

        when(mlClientService.getCropsCatalog()).thenReturn(mockCatalog);

        mockMvc.perform(get("/api/crops"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total_classes", is(3)))
                .andExpect(jsonPath("$.crops.Tomato", hasSize(3)));
    }
}
