package com.cropprotect.backend.service;

import com.cropprotect.backend.dto.GuidanceInfo;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.*;

@Service
public class GuidanceService {

    private static final Logger log = LoggerFactory.getLogger(GuidanceService.class);
    private final ObjectMapper objectMapper;
    private final Map<String, GuidanceInfo> guidanceCache = new HashMap<>();

    public GuidanceService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @PostConstruct
    public void loadGuidance() {
        try {
            ClassPathResource resource = new ClassPathResource("guidance.json");
            try (InputStream is = resource.getInputStream()) {
                Map<String, GuidanceInfo> data = objectMapper.readValue(is, new TypeReference<>() {});
                guidanceCache.putAll(data);
                log.info("[+] Loaded {} crop disease guidance protocols from guidance.json", guidanceCache.size());
            }
        } catch (Exception e) {
            log.error("[!] Failed to load guidance.json: {}", e.getMessage(), e);
        }
    }

    /**
     * Resolves preventive guidance by Crop and Disease.
     * Matches standard Crop___Disease keys.
     */
    public Optional<GuidanceInfo> getGuidance(String crop, String disease) {
        if (crop == null || disease == null) {
            return Optional.empty();
        }

        String key = crop.trim() + "___" + disease.trim();

        // 1. Exact match lookup
        if (guidanceCache.containsKey(key)) {
            return Optional.of(guidanceCache.get(key));
        }

        // 2. Case-insensitive search
        for (Map.Entry<String, GuidanceInfo> entry : guidanceCache.entrySet()) {
            if (entry.getKey().equalsIgnoreCase(key)) {
                return Optional.of(entry.getValue());
            }
        }

        // 3. Fallback for general healthy classification
        if ("healthy".equalsIgnoreCase(disease)) {
            return Optional.of(new GuidanceInfo(
                    crop + " Healthy Foliage",
                    "Healthy",
                    "None",
                    List.of("Maintain regular soil moisture and fertility balance.", "Monitor weekly for early pest ingress."),
                    "No chemical intervention required."
            ));
        }

        return Optional.empty();
    }
}
