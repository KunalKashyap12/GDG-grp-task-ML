package com.cropprotect.backend.exception;

public class MlServiceException extends RuntimeException {

    private final int statusCode;

    public MlServiceException(String message) {
        super(message);
        this.statusCode = 503;
    }

    public MlServiceException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = 503;
    }

    public MlServiceException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public int getStatusCode() {
        return statusCode;
    }
}
