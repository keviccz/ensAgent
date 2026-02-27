@echo off
REM Demo script for running Tool-Runner Agent on DLPFC_151507
REM Windows batch script

echo ============================================================
echo Tool-Runner Agent Demo - DLPFC_151507
echo ============================================================
echo.

REM Check if config exists
if not exist "..\configs\DLPFC_151507.yaml" (
    echo ERROR: Config file not found!
    echo Please ensure configs\DLPFC_151507.yaml exists
    pause
    exit /b 1
)

echo Starting pipeline...
echo.
echo Please ensure you have:
echo  1. Created the R, PY, and PY2 conda environments
echo  2. Updated the data_path in configs\DLPFC_151507.yaml
echo  3. Checked that your data directory contains all required files
echo.

pause

REM Run orchestrator
python ..\orchestrator.py --config ..\configs\DLPFC_151507.yaml

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Pipeline completed
    echo Check output\DLPFC_151507\ for results
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo ERROR: Pipeline failed
    echo Check output\DLPFC_151507\tool_runner_report.json for details
    echo ============================================================
)

pause


