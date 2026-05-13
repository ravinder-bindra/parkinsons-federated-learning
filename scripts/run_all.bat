@echo off
REM scripts\run_all.bat - Windows equivalent of run_all.sh
REM
REM Usage:
REM   scripts\run_all.bat           standard run
REM   scripts\run_all.bat --dp      with DP-SGD

setlocal
cd /d "%~dp0.."

set EXTRA=
if "%1"=="--dp" (
    set EXTRA=--dp
    echo ^>^> DP-SGD enabled
)

echo === 1/5  Preprocess (UCI Voice + NewHandPD + Telemonitoring) ===
python -m src.preprocess
if errorlevel 1 goto :error

echo.
echo === 2/5  Multi-modal federated training (15 rounds) ===
python -m src.federated_trainer --rounds 15 %EXTRA%
if errorlevel 1 goto :error

echo.
echo === 3/5  Evaluation plots ===
python -m src.evaluate
if errorlevel 1 goto :error

echo.
echo === 4/5  UPDRS regression (Telemonitoring) ===
python -m src.updrs_regression
if errorlevel 1 goto :error

echo.
echo === 5/5  Tests ===
python tests\test_pipeline.py
if errorlevel 1 goto :error

echo.
echo Done. Artefacts:
echo   results\global_model.npz
echo   results\round_history.csv
echo   results\figures\*.png
echo   results\updrs\*
echo.
echo Launch dashboard:  streamlit run app\app.py
goto :eof

:error
echo.
echo ERROR: a step failed. See the output above.
exit /b 1
