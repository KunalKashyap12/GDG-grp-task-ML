@REM ----------------------------------------------------------------------------
@REM Maven Startup Script for Windows
@REM ----------------------------------------------------------------------------
@echo off
setlocal

set "MVN_EXE=C:\Users\HP\maven\apache-maven-3.9.9\bin\mvn.cmd"
if exist "%MVN_EXE%" (
    "%MVN_EXE%" %*
) else (
    mvn %*
)
