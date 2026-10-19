package com.cropprotect.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Value("${cors.allowed-origins:*}")
    private String allowedOrigins;

    @Value("${cors.allowed-methods:GET,POST,PUT,DELETE,OPTIONS}")
    private String allowedMethods;

    @Value("${cors.allowed-headers:*}")
    private String allowedHeaders;

    @Override
    @SuppressWarnings("null")
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins(allowedOrigins != null ? allowedOrigins.split(",") : new String[]{"*"})
                .allowedMethods(allowedMethods != null ? allowedMethods.split(",") : new String[]{"GET", "POST", "PUT", "DELETE", "OPTIONS"})
                .allowedHeaders(allowedHeaders != null ? allowedHeaders.split(",") : new String[]{"*"})
                .maxAge(3600);
    }
}
