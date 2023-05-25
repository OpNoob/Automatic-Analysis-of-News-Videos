CALL venv/Scripts/activate.bat

pyinstaller --name=NewsVideoAnalysis --onefile --windowed --hidden-import face_recognition --add-data "venv/Lib/site-packages/face_recognition_models/models;face_recognition_models/models" --add-data "Tesseract-OCR;Tesseract-OCR" --add-data "assets;assets" --icon=assets/2400ppi.ico gui.py

PAUSE