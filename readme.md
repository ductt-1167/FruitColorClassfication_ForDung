# Dataset:
- We using data from camera captures fruits with white background
# Before running project, you need:
1. Install Python 3

2. Setup interpreter: Goto Pycharm --> File --> Settings --> Project:FruitClassification --> Python interpreter --> Add
    --> Virtualenv environment --> new environment --> Ok 
    
4. Install needed packages: I list packages which used in project in setup.txt. Doing run :  pip install -r setup.txt in terminal 

# Run project
## - For training model (Just run this file when you want to re-train model) (Not recommended)
- trainning.py

## - For using model to classification 
- main.py: turn on camera and capture fruit images, then classify them 
- test.py: test model with folder image 
