@echo off
REM quick_lotus_fix.bat - Quick fix for Lotus environment on Windows

echo Fixing Lotus environment...

REM Activate lotus environment
call .lotus_env\Scripts\activate.bat

REM Install missing packages
echo Installing matplotlib...
pip install matplotlib

echo Installing plotly...
pip install plotly

REM Test the environment
echo Testing environment...
python -c "import pandas, numpy, requests, matplotlib, plotly; print('Environment ready')"

echo Done! Try your query again.
pause@echo off
REM quick_lotus_fix.bat - Quick fix for Lotus environment on Windows

echo Fixing Lotus environment...

REM Activate lotus environment
call .lotus_env\Scripts\activate.bat

REM Install missing packages
echo Installing matplotlib...
pip install matplotlib

echo Installing plotly...
pip install plotly

REM Test the environment
echo Testing environment...
python -c "import pandas, numpy, requests, matplotlib, plotly; print('Environment ready')"

echo Done! Try your query again.
pause