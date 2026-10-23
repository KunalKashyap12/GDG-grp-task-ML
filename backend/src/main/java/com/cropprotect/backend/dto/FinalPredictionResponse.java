package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class FinalPredictionResponse {

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

    @JsonProperty("guidance")
    private GuidanceInfo guidance;

    @JsonProperty("disclaimer")
    private String disclaimer = "Notice: This automated recommendation provides non-binding agricultural decision support. It is not an agronomist-confirmed clinical diagnosis. Verify physical symptoms in-field before purchasing chemical inputs.";

    @JsonProperty("timestamp")
    private Instant timestamp = Instant.now();

    public FinalPredictionResponse() {
    }

    public static FinalPredictionResponse from(MlPredictionResponse mlResponse, GuidanceInfo guidance) {
        FinalPredictionResponse res = new FinalPredictionResponse();
        res.setStatus(mlResponse.getStatus());
        res.setCrop(mlResponse.getCrop());
        res.setDisease(mlResponse.getDisease());
        res.setConfidence(mlResponse.getConfidence());
        res.setCertainty(mlResponse.getCertainty());
        res.setTop3(mlResponse.getTop3());
        res.setQuality(mlResponse.getQuality());
        res.setMessage(mlResponse.getMessage());
        res.setGuidance(guidance);
        res.setTimestamp(Instant.now());
        return res;
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

    public GuidanceInfo getGuidance() {
        return guidance;
    }

    public void setGuidance(GuidanceInfo guidance) {
        this.guidance = guidance;
    }

    public String getDisclaimer() {
        return disclaimer;
    }

    public void setDisclaimer(String disclaimer) {
        this.disclaimer = disclaimer;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Instant timestamp) {
        this.timestamp = timestamp;
    }
}
