@echo off
call "C:\Users\florian1911\AppData\Local\anaconda3\Scripts\activate.bat" mystreamlit
cd /d "C:\Users\florian1911\Documents\Streamlit\Loudness"
streamlit run "Loudness.py"
pause