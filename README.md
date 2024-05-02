# Machine Learning Maze Solver
## Usage
1. Install the required dependencies.
   ```
   pip install -r requirements.txt
   ```
2. Run the app.
   ```
   python app.py
   ```
### Train Model
This project contains a pre-trained model for classifying maze images in directory `pytorch_maze_classifier`. However, if you would like to re-train the model or adjust any parameters, run the following command: 
```bash
python model_creator.py
```
### Test Model
To test the accuracy of the model on Windows run: 
```
python maze_classifier.py
```
Note that if the model file does not exist in directory `pytorch_maze_classifier` this will not work.
