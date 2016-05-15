import data_preparation_tools as dpt
import features_generation_tools as fgt
from sklearn.externals import joblib

DEFAULT_VERSION = "1.1"

class ScoringModel:
    # TODO: add the ability to pass in the array of transformations as well...for now we just use eveyrthing
    def __init__(self, features_generator, ml_model, transformations=dpt.TRANSFORMATION_STEPS, version=DEFAULT_VERSION):
        self.transformations = transformations
        self.features_generator = features_generator
        self.ml_model = ml_model
        self.version = version

    def score(self, text, context):
        transformed = dpt.run_transformations_on_single_sentence(text, context,self.transformations)
        features = self.features_generator.transform([transformed])
        return self.ml_model.predict_proba(features)[0]

    def save_model(self, file_path):
        return joblib.dump(self,file_path, compress=True)

    @staticmethod
    def from_file(file_path):
        return joblib.load(file_path)