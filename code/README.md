### Running through colab files
This folder contains colab notebooks(.ipynb files) for each of the model used.

#### Dependencies

The Google Colab’s Pro GPU (NVIDIA K80/P100) was employed for fine-tuning and testing the algorithms. Training of each model takes about 6 hours to finish. No hyperparameter tuning was used during training. Following parameters were set during the model's training:

 Epochs=3, learning rate=2e-5, batch size=10, optimizer = AdamW( weight decay of 0.01)

Default values were used for other parameters. 

However, since we are using the checkpoints, even Google Colab free edition will also suffice

#### Instructions

1. Get the folder 'checkpoint-2000' in your google drive from the link shared below for each of the model.
     * Roberta-checkpoint - [https://drive.google.com/drive/folders/10L0uZNR0o3w99iqeUgamVdlpeub0DF6w?usp=sharing](https://drive.google.com/drive/folders/10L0uZNR0o3w99iqeUgamVdlpeub0DF6w?usp=sharing)
     * Distilbert-checkpoint - [https://drive.google.com/drive/folders/1nDeIZKgDxahMu9tAXXMCNGLIh_13wUl_?usp=sharing](https://drive.google.com/drive/folders/1nDeIZKgDxahMu9tAXXMCNGLIh_13wUl_?usp=sharing)
2. Right click on the folder 'checkpoint-2000' and add shortcut to 'MyDrive'
2. Copy the path of the folder where you copied the folder, for example '/content/drive/MyDrive/checkpoint-2000', in the second cell of the colab notebook to initialize variable 'model_path'
3.  Run all the cells to see the output.
4.  You will be prompted to authorize the google drive. Please authorize for running the file.

### Running through python files
#### Download the checkpoint-2000 folder for the model to run.
 
#### Setup
```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# change directory to the repo where we have requirements file
$ cd f2021-proj-ravikanagpal/code/

# Install python dependencies
$ pip3 install  -r requirements.txt 
```
#### Use below command to run the models

   `python3 main.py --checkpoint <Absolute path to checkpoint-2000 directory for distilbert> --model distilbert-base-uncased --output <Absolute path to output text file>
   
   `python3 main.py --checkpoint <Absolute path to checkpoint-2000 directory for roberta> --model RobertaTokenizerFast --output <Absolute path to output text file>
   
   For example:
   
   `python3 main.py --checkpoint ./distilled_base_model_checkpoint_2000/ --model distilbert-base-uncased --output ./output/distil_result.txt`
   
   `python3 main.py --checkpoint ./roberta_model_checkpoint_2000/ --model RobertaTokenizerFast --output ./output/roberta_result.txt`
   
#### This command will print the span predictions in text file and f1 score will be printed on the screen.

### Please note that F1 score could be less than the ones reported in the report file as we are using checkpoint-2000 which is less than the total number of iterations in epochs 3. Since, this was the last and the nearest checkpoint, we have shared this.
