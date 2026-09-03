@echo off
REM Sets up the fraud detection project on Windows (section 4.1: Windows 11,
REM Python 3, Visual Studio Code). Run this once from the project root.

echo Creating folders...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete.
echo.
echo Next steps:
echo   1. Download creditcard.csv from Kaggle and place it in data\  (see data\README.md)
echo   2. python -m src.data_prep
echo   3. python -m src.evaluate
echo   4. python -m src.make_samples
echo   5. set FLASK_SECRET_KEY=change-this-in-production
echo   6. python webapp\app.py
echo.
pause
