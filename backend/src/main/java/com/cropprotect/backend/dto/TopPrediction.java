package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TopPrediction {

    @JsonProperty("label")
    private String label;

    @JsonProperty("score")
    private Double score;

    public TopPrediction() {
    }

    public TopPrediction(String label, Double score) {
        this.label = label;
        this.score = score;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }
}
