# Gait_Recognition
Repository for major project in gait recognition

## Training in Google colab
- Step 1 : Download this repo as a zip file
- Step 2 : Create a new notebook in google colab, connect with T4 GPU (for faster training)
- Step 3 : Upload the zip on google colab
- Step 4 : In the ipynb file in google colab, create a new cell and paste these commands  and run
```
!unzip /content/Gait_Recognition-main.zip
!gdown 1EjEb53EEK9RZKND5dfF8TvoCrFUxqC7r -O /content/data.zip
!mv /content/Gait_Recognition-main /content/Code
!mkdir /content/Code/data
!unzip /content/data.zip -d /content/Code/data

```
These commands will 1. unzip the code, 2. download the dataset from gdrive 3. move and create required directories

- Step 5 : Create a new cell to mount the google drive to save the checkpoints, just put the below code, run the cell, give the permissions
```
from google.colab import drive
drive.mount('/content/drive')

```

- Step 6 : Double click on code/config.py file, it will open it in a editor in collab itself (on right side), modify  the config object according to requirement and save the file (Ctrl + s). Please set the ```base_dir : '/content/Code' ```
- Step 7 : Create another new cell and run ```!python /content/Code/preprocessing.py ```, this will preprocess the data and save it in the specified directory. please note that the intermediate data will be deleted to avoid disk space issues.
- Step 8 : Create another new cell and run ```!python /content/Code/training_and_eval.py```, this will train the model according to the config file, and test it on the test sample and display the results as well.


## Training in Kaggle
- Step 1 : Download this repo as a zip file
- Step 2 : Create a new notebook in Kaggle, connect with GPU (for faster training)
- Step 3 : Upload the zip as the dataset and wait for few seconds to get it processed.
- Step 4 : In the ipynb file in google colab, create a new cell and paste these commands  and run
```
import sys

!cp -r /kaggle/input/gait-final-v3/Gait_Recognition-main /kaggle/working/
!mv /kaggle/working/Gait_Recognition-main /kaggle/working/Code
sys.path.insert(1, '/kaggle/working/Code')

!gdown 1EjEb53EEK9RZKND5dfF8TvoCrFUxqC7r -O /content/data.zip
!mkdir /kaggle/working/Code/data
!unzip -q /content/data.zip -d /kaggle/working/Code/data
!rm -rf /content/data.zip

```
These commands will 1. unzip the code, 2. download the dataset from gdrive 3. move and create required directories 4.Add code path (as we can't directly run the file by python command, we make the whole code as a module and import it, that's why we need a path to the code in the system.)

- Step 5 : Create a new cell and create the config object, (Details about config object will be described later)
for example
```
config = {
    
    'base_dir': '/kaggle/working/Gait_Recognition-main', #for colab : '/content/Code'
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    'gait_features'
                    ],

    'drive_checkpoint_path' : '/kaggle/working/trained_gait_model_checkpoints',

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        'rope' : False,
        'contrastive' : False,
        'k_fold' : False,
        'epochs' : 60
    },

}

```

- Step 6 : Please set the ```base_dir : '/kaggle/working/Gait_Recognition-for_kaggle' ``` or according to your naming conventions
- Step 6 : Create another new cell and run
```
from preprocessing import preprocessor
preprocessor(config)
```
this will preprocess the data and save it in the specified directory. please note that the intermediate data will be deleted to avoid disk space issues.
- Step 8 : Create another new cell and run
```
from training_and_eval import train_and_eval
train_and_eval(config)
```
this will train the model according to the config object, and test it on the test sample and display the results as well.

Note: If the dataset size is huge and RAM is overflowing while training, do these:
      After the preprocessing is completed, Goto Run > Restart and clear cell outputs, this will free up the RAM. 
      After that create a new cell and add the path in the system and define config object then start the training, for example
  ```
      import sys
      sys.path.insert(1, '/kaggle/working/Gait_Recognition-for_kaggle')

    config = {
    
    'base_dir': '/kaggle/working/Gait_Recognition-main', 
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    'gait_features'
                    ],

    'drive_checkpoint_path' : '/kaggle/working/trained_gait_model_checkpoints',

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        'rope' : False,
        'contrastive' : False,
        'k_fold' : False,
        'epochs' : 60
    },

}
```


## Config object

The config object looks like
```
config = {
    
    'base_dir': '/kaggle/working/Code', #for colab : '/content/Code'
    # 'last_preprocessing' : 'gait_cycles_iigc',
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    #gait_cycles_iigc,
                    'gait_features',
                    'event_features'],

    'drive_checkpoint_path' : '/content/drive/My Drive/trained_gait_model_checkpoints',

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        # 'max_len' : 2048,
        # 'd_model : 256,
        'rope' : False,
        'contrastive' : False,
        # 'contrastive_weight' : 0.5,
        'k_fold' : False,
        'epochs' : 60,
    #    'cls_head_hidden_layers': [256, 128]
    },

}
```

1. base_dir : It's the path of the code directory, this is used to access the data input and output during preprocessing and training. Above example shows, if my complete code folder is in kaggle/working/ directory named as Gait_Recognition-main
2. preprocess : It will take input as a list of strings, each defining various preprocessing steps. Please make sure you put them in correct order, as according to the list order the data will be preprocessed.
3. drive_checkpoint_path : path to the directory where the checkpoints are saved and where it the checkpoints will be searched for resuming.
4. training : This is a dictionary representing the training oriented parameters.
