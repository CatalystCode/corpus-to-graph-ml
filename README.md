# corpus-to-graph-ml
This repository contains machine learning related work for the corpus to graph project, including Jupyter research notebooks and a Flask webservice to host the model.

The packages folder contains python code with the main logic for the data transformation and the feature generation tools.

 The webservice contains an example for a flask based scoring service that can be used in order to expose the trained model.

The data_preparation notebook contains an example of running the data transformation pipeline, and the features_classification notebook contains code examples for generating different features and training and evaluating different classifiers.

The only missing piece that shold be provided is an entity recognition endpoint (specifically here we used [GNAT](http://gnat.sourceforge.net/)). You can also alternatively provide a text file with the results of the entity recognition process.
