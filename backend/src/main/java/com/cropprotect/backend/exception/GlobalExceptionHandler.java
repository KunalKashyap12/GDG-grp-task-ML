package com.cropprotect.backend.exception;

import com.cropprotect.backend.dto.ErrorResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(MlServiceException.class)
    public ResponseEntity<ErrorResponse> handleMlServiceException(MlServiceException ex) {
        log.error("ML service communication failed: {}", ex.getMessage());
        HttpStatus status = ex.getStatusCode() == 504 ? HttpStatus.GATEWAY_TIMEOUT : HttpStatus.SERVICE_UNAVAILABLE;
        ErrorResponse err = new ErrorResponse(
                "ML_SERVICE_UNAVAILABLE",
                ex.getMessage(),
                status.value()
        );
        return ResponseEntity.status(status).body(err);
    }

    @ExceptionHandler(InvalidImageException.class)
    public ResponseEntity<ErrorResponse> handleInvalidImageException(InvalidImageException ex) {
        log.warn("Invalid image upload request: {}", ex.getMessage());
        ErrorResponse err = new ErrorResponse(
                "INVALID_IMAGE_PAYLOAD",
                ex.getMessage(),
                HttpStatus.BAD_REQUEST.value()
        );
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(err);
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ErrorResponse> handleMaxSizeException(MaxUploadSizeExceededException ex) {
        log.warn("Image upload size limit exceeded: {}", ex.getMessage());
        ErrorResponse err = new ErrorResponse(
                "IMAGE_SIZE_EXCEEDED",
                "Uploaded image exceeds maximum allowable limit of 5 MB.",
                HttpStatus.PAYLOAD_TOO_LARGE.value()
        );
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(err);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneralException(Exception ex) {
        log.error("Unexpected internal server error occurred: ", ex);
        ErrorResponse err = new ErrorResponse(
                "INTERNAL_SERVER_ERROR",
                "An unexpected server error occurred: " + ex.getMessage(),
                HttpStatus.INTERNAL_SERVER_ERROR.value()
        );
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(err);
    }
}
