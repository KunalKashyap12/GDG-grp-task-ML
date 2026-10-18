package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class QualityInfo {

    @JsonProperty("blur_score")
    private Double blurScore;

    @JsonProperty("brightness")
    private Double brightness;

    public QualityInfo() {
    }

    public QualityInfo(Double blurScore, Double brightness) {
        this.blurScore = blurScore;
        this.brightness = brightness;
    }

    public Double getBlurScore() {
        return blurScore;
    }

    public void setBlurScore(Double blurScore) {
        this.blurScore = blurScore;
    }

    public Double getBrightness() {
        return brightness;
    }

    public void setBrightness(Double brightness) {
        this.brightness = brightness;
    }
}
