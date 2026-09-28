@echo off
title VeriShield AI - Multimodal Fake News Detection System
cd /d "%~dp0"
echo ======================================================================
echo  Starting VeriShield AI Application...
echo ======================================================================
python run.py
if errorlevel 1 (
    echo.
    echo An unexpected error occurred.
    pause
)
