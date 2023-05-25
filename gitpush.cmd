git init

git lfs install
git lfs track Tesseract-OCR/libtesseract-5.dll
@REM git lfs track "*.mp4"
@REM git lfs track "*.pkl"
@REM git lfs track "*.dll"

git add .
git commit -m "Uploading folder"
git remote add origin https://github.com/OpNoob/Automatic-Analysisof-News-Videos.git
git checkout -b development_tests
git branch
git push origin development_tests

PAUSE
