@echo off
:: Navigate to the directory (using /d in case it's on a different drive)
cd /d "C:\Github\AnalyzeBig\Equity_Finder"

:: Optional: If you use a virtual environment, uncomment the next line:
:: call venv\Scripts\activate

:: Run the Streamlit app
streamlit run Equity_Evaluator.py

:: Keeps the window open if the app crashes so you can see the error
pause