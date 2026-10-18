package com.cropprotect.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CropsCatalogResponse {

    @JsonProperty("crops")
    private Map<String, List<String>> crops = new HashMap<>();

    @JsonProperty("total_classes")
    private Integer totalClasses;

    @JsonProperty("classes_list")
    private List<String> classesList;

    public CropsCatalogResponse() {
    }

    public CropsCatalogResponse(Map<String, List<String>> crops, Integer totalClasses, List<String> classesList) {
        this.crops = crops;
        this.totalClasses = totalClasses;
        this.classesList = classesList;
    }

    public Map<String, List<String>> getCrops() {
        return crops;
    }

    public void setCrops(Map<String, List<String>> crops) {
        this.crops = crops;
    }

    public Integer getTotalClasses() {
        return totalClasses;
    }

    public void setTotalClasses(Integer totalClasses) {
        this.totalClasses = totalClasses;
    }

    public List<String> getClassesList() {
        return classesList;
    }

    public void setClassesList(List<String> classesList) {
        this.classesList = classesList;
    }
}
