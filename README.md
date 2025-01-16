# Gait_Recognition
Repository for major project in gait recognition

## Training
- Step 1 : download the zip file
- Step 2 : upload the zip on google colab
- Step 3 : in the ipynb file in google colab, create a new cell and paste these commands  and run
```
!unzip /content/Gait_Recognition-main.zip
!gdown 1EjEb53EEK9RZKND5dfF8TvoCrFUxqC7r -O /content/data.zip
!mv /content/Gait_Recognition-main /content/Code
!mkdir /content/Code/data
!unzip /content/data.zip -d /content/Code/data

```
These commands will 1. unzip the code, 2. download the dataset from gdrive 3. move and create required directories

- Step 4 : Double click on code/main.py file, it will open it in a editor in collab itself (on right side), modify  the config object according to requirement and save the file (Ctrl + s)

- Step 5 : Create another new cell and run ```!python /content/Code/main.py```
