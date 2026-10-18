package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.ArrayList;
import java.util.List;

public class MlPredictionResponse {

    @JsonProperty("status")
    private String status;

    @JsonProperty("crop")
    private String crop;

    @JsonProperty("disease")
    private String disease;

    @JsonProperty("confidence")
    private Double confidence;

    @JsonProperty("certainty")
    private String certainty;

    @JsonProperty("top3")
    private List<TopPrediction> top3 = new ArrayList<>();

    @JsonProperty("quality")
    private QualityInfo quality;

    @JsonProperty("message")
    private String message;

    public MlPredictionResponse() {
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getCrop() {
        return crop;
    }

    public void setCrop(String crop) {
        this.crop = crop;
    }

    public String getDisease() {
        return disease;
    }

    public void setDisease(String disease) {
        this.disease = disease;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getCertainty() {
        return certainty;
    }

    public void setCertainty(String certainty) {
        this.certainty = certainty;
    }

    public List<TopPrediction> getTop3() {
        return top3;
    }

    public void setTop3(List<TopPrediction> top3) {
        this.top3 = top3;
    }

    public QualityInfo getQuality() {
        return quality;
    }

    public void setQuality(QualityInfo quality) {
        this.quality = quality;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}
