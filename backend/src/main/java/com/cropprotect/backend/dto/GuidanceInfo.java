package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.ArrayList;
import java.util.List;

public class GuidanceInfo {

    @JsonProperty("disease_name")
    private String diseaseName;

    @JsonProperty("category")
    private String category;

    @JsonProperty("urgency")
    private String urgency;

    @JsonProperty("preventive_measures")
    private List<String> preventiveMeasures = new ArrayList<>();

    @JsonProperty("treatment_summary")
    private String treatmentSummary;

    public GuidanceInfo() {
    }

    public GuidanceInfo(String diseaseName, String category, String urgency, List<String> preventiveMeasures, String treatmentSummary) {
        this.diseaseName = diseaseName;
        this.category = category;
        this.urgency = urgency;
        this.preventiveMeasures = preventiveMeasures;
        this.treatmentSummary = treatmentSummary;
    }

    public String getDiseaseName() {
        return diseaseName;
    }

    public void setDiseaseName(String diseaseName) {
        this.diseaseName = diseaseName;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getUrgency() {
        return urgency;
    }

    public void setUrgency(String urgency) {
        this.urgency = urgency;
    }

    public List<String> getPreventiveMeasures() {
        return preventiveMeasures;
    }

    public void setPreventiveMeasures(List<String> preventiveMeasures) {
        this.preventiveMeasures = preventiveMeasures;
    }

    public String getTreatmentSummary() {
        return treatmentSummary;
    }

    public void setTreatmentSummary(String treatmentSummary) {
        this.treatmentSummary = treatmentSummary;
    }
}
