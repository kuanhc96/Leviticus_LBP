from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import classification_report
from imutils import paths
from scipy.stats import uniform
from toolbox.utils.compare_directories import _isEqualSubDirs

from toolbox.tf.localbinarypatterns import LocalBinaryPatterns
from config import PKL_PATH

import numpy as np
import cv2
import os
import pickle


# Sample cURL:
# curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"trainOnly": true,"dataset":"/app/texture_dataset","numPoints":24, "radius":8}'
app = FastAPI()

class LBPPredictRequest(BaseModel):
    trainTaskId: str
    trainDataset: str # required
    predictDataset: str # required
    # numPoints & radius are LBP specific parameters, used to generate
    # histograms for the SVC to train on 
    # These values need to be provided by the master node and need to match
    # The values that were used during initial training
    numPoints: Optional[int] = 24
    radius: Optional[int] = 8

class LBPTrainRequest(BaseModel):
    # this is a value auto-generated by the master node
    # It represents the ID of the current training job
    taskId: str 
    # This value will determite if a testing set is needed to be set aside 
    # for scoring. By default it is False, which means a test set should 
    # be set aside for scoring
    trainOnly: Optional[bool] = False
    # numPoints & radius are LBP specific parameters, used to generate
    # histograms for the SVC to train on 
    numPoints: Optional[int] = 24
    radius: Optional[int] = 8
    # C is a parameter specific to the SVC and it is a measure of "strictness"
    # C must be strictly positive
    # Need to restrict client to only submitting positive values in the master node
    # If C = 0.0, that means the user did not input a value for C,
    # which will trigger hyperparameter search to find the best value for C
    # If C is provided, it is assumed that it will be > 0.0,
    # in which case hyperparameter search will not be called and the provided
    # C value will simply be used for training
    C: Optional[float] = 0.0
    # This is a string that represents the path to the training data
    dataset: str

class LBPTrainResponse(BaseModel):
    # This is the same taskId that was sent to this container by the master node
    taskId: str
    # This is the path to which the pickled model is saved
    modelPath: str
    # This is the score representing the model's performance
    accuracy: float
    # This value is either the best C found through hyperparameter search,
    # or it is simply the original C value sent from the master node
    C: float

    """SVC parameters"""
    class_weight: float
    dual: str
    fit_intercept: bool
    intercept_scaling: int
    loss: str
    max_iter: int
    multi_class: str
    penalty: str
    random_state: int
    tol: float
    verbose: int
    """SVC parameters"""

    # This is a long string representing the classification report 
    # of the resulting model
    classificationReport: str


# The train request for Local Binary Patterns
# The structure of the dataset is assumed to be
# dataset/{label}/*.png
# If trainOnly = True, then no need to prepare a
# test set, and the training data can simply be
# used for scoring
# Otherwise, 10% of the data will be set aside for
# testing
@app.post("/train")
def train(request: LBPTrainRequest) -> dict:
    print("[INFO] Received Local Binary Pattern Training Request")
    # initialize the local binary patterns descriptor along with the data and label lists
    dataset = request.dataset
    print(f"[INFO] Dataset Received For Training: {dataset.split(os.path.sep)[-1]}")
    taskId = request.taskId
    trainOnly = request.trainOnly
    numPoints = request.numPoints
    C_value = request.C
    radius = request.radius
    trainX = []
    testX = []

    trainLabels = []
    testLabels = []

    predictions = []

    desc = LocalBinaryPatterns(numPoints, radius)

    imagePaths = list(paths.list_images(dataset))
    print("[INFO] Preparing Training Data")
    if trainOnly:
        trainPaths = imagePaths
        # no need to set aside a test set
        testPaths = []
    else:
        (trainPaths, testPaths) = train_test_split(imagePaths, test_size=0.1)

    # Load each image in the trainPaths
    for trainPath in trainPaths:
        image = cv2.imread(trainPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get histogram using the LBP algorithm
        hist = desc.describe(gray)

        # The traingLabels (trainY) are obtained from the path names
        trainLabels.append(trainPath.split(os.path.sep)[-2])
        # The histograms of the images will form the actual training set
        trainX.append(hist)

    print("[INFO] Fitting Model")
    if C_value == 0.0: # This is actually an illegal value, meaning it was not inputted
        param_grid = {
            # hyperparameter search will look for values of C between 1.0 and 100.0
            'C': uniform(loc=1.0, scale=100.0)
        }
        model = LinearSVC()
        randomizedSearch = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10,
                scoring='accuracy',
                cv=RepeatedKFold(
                    n_splits=10, 
                    n_repeats=3,
                    random_state=42
                )
            )
        randomizedSearch.fit(trainX, trainLabels)
        C = randomizedSearch.best_params_['C']
        print(f"[DEBUG] Best C parameter for the model is found to be {C}")
        model = randomizedSearch.best_estimator_
    else:
        # Since the C value was provided, simply use if for training
        model = LinearSVC(C=C_value)
        model.fit(trainX, trainLabels)
    print("[INFO] Model Fitting Complete")

    if trainOnly:
        print("[INFO] Testing Not Required. Proceeding To Response Preparation")
        testX = trainX
        testLabels = trainLabels
        # get predictions of the training set to use for the classification report
        predictions = model.predict(np.array( testX ))
        uniqueLabels = np.unique(trainLabels)

    else:
        print("[INFO] Preparing Testing Data")

        # Similar to the training set, load each image and perform a 
        # prediction on them to see what the model thinks it is
        # This prediction will be recorded for scoring later
        for testPath in testPaths:
            testLabels.append(testPath.split(os.path.sep)[-2])
            image = cv2.imread(testPath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            testX.append(hist)
            prediction = model.predict(hist.reshape(1, -1))[0]
            predictions.append(prediction)
    
        uniqueLabels = np.unique(np.concatenate(( trainLabels, testLabels )))

    print("[INFO] Scoring Model")
    accuracy = model.score(testX, testLabels)
    classificationReport = classification_report(testLabels, predictions, labels=uniqueLabels)
    print("[INFO] Train Request Complete, Returning Training Results")
    print("[INFO] Saving Trained Model")
    modelPath = os.path.join(PKL_PATH, taskId + ".pkl")
    with open(modelPath, 'wb') as f:
        pickle.dump(model, f)
    print("[INFO] Training Model Saved")


    return {"taskId": taskId, "modelPath": modelPath, "accuracy": accuracy, 
            **model.get_params(), "classificationReport": classificationReport}

# If the directory for predict has the same structure of
# the directory that was provided for training, then the
# data is organized in a way for ground-truth labels to be
# inferred, which means a score for prediction can be returned;
# Otherwise, the ground-truth cannot be inferred, and the 
# results of the prediction can simply be returned
@app.post("/predict")
def predict(request: LBPPredictRequest) -> dict:
    # obtain request parameters
    trainDataset = request.trainDataset
    predictDataset = request.predictDataset
    trainTaskId = request.trainTaskId
    numPoints = request.numPoints
    radius = request.radius

    labels = []
    testX = []
    imageNames = []

    # check if predict directory has the same structure as the train directory
    isEqualSubDirs = _isEqualSubDirs(trainDataset, predictDataset)
    # check if the directory is a series of images with no directory
    isDirOfImages = len(next(os.walk(predictDataset))[1]) == 0
    # The predict directory has to be one of the above
    if not isEqualSubDirs and not isDirOfImages:
        # there is an error in the input directory
        return {"error": "Directory mismatch - incorrect number of subdirectories"}

    # saved pickled model
    with open(os.path.join(PKL_PATH, trainTaskId + ".pkl"), 'rb') as f:
        savedModel = pickle.load(f)

    # get list of images
    imagePaths = list(paths.list_images(predictDataset))

    # initialize LBP algorithm
    desc = LocalBinaryPatterns(numPoints, radius)

    for imagePath in imagePaths:
        # load image
        image = cv2.imread(imagePath)
        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get histogram using the LBP algorithm
        hist = desc.describe(gray)

        # The labels are obtained from the path names
        labels.append(imagePath.split(os.path.sep)[-2])
        # The image names are obtained from the path names
        imageName = os.path.join(imagePath.split(os.path.sep)[-2], imagePath.split(os.path.sep)[-1]) 
        imageNames.append(imageName)
        # The histograms of the images will form the actual training set
        testX.append(hist)

    testX = np.array(testX)

    # make predictions
    predictions = savedModel.predict(testX)

    if isEqualSubDirs:
        # perform prediction on the images and use the directories as ground truth
        uniqueLabels = np.unique(labels)

        accuracy = savedModel.score(testX, labels)
        report = classification_report(labels, predictions, labels=uniqueLabels)
        return {
            "accuracy": accuracy,
            "classificationReport": report, 
            "predictions": dict(zip(imageNames, predictions.tolist()))
            }

    elif isDirOfImages:
        # This is a single directory of images; 
        # perform the prediction without regards to the ground truth
        accuracy = None
        report = None
        return {
            "accuracy": None,
            "classificationReport": None,
            "predictions": dict(zip(imageNames, predictions.tolist()))
        }
