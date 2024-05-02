# Machine Learning Maze Solver

## Usage
Before running any of the code make sure to install the dependencies: <code>pip install -r requirements.txt</code>
### Train Model
This project contains a pre-trained model for classifying maze images in directory pytorch_maze_classifier. However, if you would like to re-train the model or adjust any parameters, run/edit model_creator.py.
### Test Model
To test the accuracy of the model on Windows run: <code>python maze_classifier.py</code> and on Mac run: <code>python3 maze_classifier.py</code>. Note that if the model file does not exist in directory pytorch_maze_classifier this will not work.
### Run App
To run the actual app on Windows run: <code>python app.py</code> and on Mac run: <code>python3 app.py</code>. Note that sample maze and non-maze inputs are suplied in directory sample_inputs. Alternatively, you can upload your own image or click the "Generate Maze" button to generate a sample maze image.
