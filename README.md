# guitar-classification

To perform image classification on guitar images, this solution takes two approaches.  In one approach, we take color histograms of the RGB channels in the training images to develop our feature vectors.  The code to create the histograms is in `colorHistograms.py`, and the resulting CSV file is uploaded as a dataset into Azure Machine Learning.  The experiment workflow is shown in `azureML_experiment.jpg`.

In the second approach, we use Keras with a CNTK backend to implement a ConvNet for classification.  The CNN code is in `train.py`.
