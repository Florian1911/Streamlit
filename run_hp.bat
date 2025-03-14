@echo off
call "C:\Users\flori\anaconda3\Scripts\activate.bat" env
cd /d "C:\Users\flori\Documents\Streamlit\Loudness"
streamlit run "Loudness.py"
pause