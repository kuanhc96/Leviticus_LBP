from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from toolbox.tf.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import classification_report
from imutils import paths
from scipy.stats import uniform
import numpy as np
import cv2
import os

# Sample cURL:
# curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"trainOnly": true,"dataset":"/app/texture_dataset","numPoints":24, "radius":8}'
app = FastAPI()

class LBPPredictRequest(BaseModel):
    dummy: str

class LBPTrainRequest(BaseModel):
    trainOnly: Optional[bool] = False
    numPoints: Optional[int] = 24
    radius: Optional[int] = 8
    # C must be strictly positive
    # Need to restrict client to only submitting positive values
    C: Optional[float] = 0.0
    dataset: str

class LBPTrainResponse(BaseModel):
    accuracy: float
    C: float
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
    classificationReport: str


@app.post("/train")
def train(request: LBPTrainRequest) -> dict:
    print("[INFO] Received Local Binary Pattern Training Request")
    # imitialize the local binary patterns descriptor along with the data and label lists
    dataset = request.dataset
    print(f"[INFO] Dataset Received For Training: {dataset.split(os.path.sep)[-1]}")
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
        testPaths = []
    else:
        (trainPaths, testPaths) = train_test_split(imagePaths, test_size=0.1)

    for trainPath in trainPaths:
        image = cv2.imread(trainPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)

        trainLabels.append(trainPath.split(os.path.sep)[-2])
        trainX.append(hist)

    print("[INFO] Fitting Model")
    if C_value == 0.0: # This is actually an illegal value, meaning it was not inputted
        param_grid = {
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
        model = LinearSVC(C=C_value)
        model.fit(trainX, trainLabels)
    print("[INFO] Model Fitting Complete")

    if trainOnly:
        print("[INFO] Testing Not Required. Proceeding To Response Preparation")
        testX = trainX
        testLabels = trainLabels
        predictions = model.predict(np.array( testX ))
        uniqueLabels = np.unique(trainLabels)

    else:
        print("[INFO] Preparing Testing Data")

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

    return {"accuracy": accuracy, **model.get_params(), "classificationReport": classificationReport}

def predict(request: LBPPredictRequest) -> dict:
    pass
