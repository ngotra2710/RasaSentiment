import rasa
from rasa.nlu.components import Component
import rasa.utils.io as utils_io
from rasa.nlu.model import Metadata

import nltk
from nltk.classify import NaiveBayesClassifier
import os

import typing
from typing import Any, Optional, Text, Dict
import jsonpickle
from sklearn.svm import SVC
import json
from json import JSONEncoder
import numpy

SENTIMENT_MODEL_FILE_NAME = "sentiment_classifier.pkl"

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class SentimentAnalyzer(Component):
    """A custom sentiment analysis component"""
    name = "sentiment"
    provides = ["entities"]
    requires = ["tokens"]
    defaults = {}
    language_list = ["en"]
    print('initialised the class')

    def __init__(self, component_config=None):
        super(SentimentAnalyzer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""
        with open('labels.txt', 'r') as f:
            labels = f.read().splitlines()
        training_data = training_data.training_examples #list of Message objects
        tokens = []
        features = []
        for t in training_data:
            print("-----------------------------------------------")
            # print(t.__dict__.keys())
            # print(t.data)
            # print(t.features)            
            if t.data.get('text_tokens') == None or len(t.features) == 0:
                continue               
            tokens.append(map(lambda x: x.text, t.data.get('text_tokens')))
            print(t.data.get('text_tokens'))
            for f in t.features:
                print(f.__dict__.keys())  
                print(f.features)  
                print(f.type)
                print(f.origin)
                print(f.attribute)
                print(f.features.shape) 
                if f.type == "sentence": 
                    features.append(f.features[0,:])  
            # print(tokens)
        # print("features: ", features)
        processed_tokens = [self.preprocessing(t) for t in tokens]
        print("processed_tokens ", processed_tokens)
        labeled_data = [(t, x) for t,x in zip(processed_tokens, labels)]
        # labeled_data = [(t, f, x) for t,f,x in zip(processed_tokens, features, labels)]
        print("labeled_data ", labeled_data)
        self.clf = NaiveBayesClassifier.train(labeled_data)
        self.clf1 = SVC(kernel="linear")
        self.clf1.fit(features, labels)


    def convert_to_rasa(self, value, confidence, type_predict):
        """Convert model output into the Rasa NLU compatible output format."""
        # print("+++++++++++++++++++")
        # print(confidence)
        entity = {"value": value,
                  "confidence": confidence,
                  "entity": type_predict,
                  "extractor": type_predict + "_extractor"}
        # print(entity)
        return entity

    def preprocessing(self, tokens):
        """Create bag-of-words representation of the training examples."""

        return ({word: True for word in tokens})

    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""

        if not self.clf:
            # component is either not trained or didn't
            # receive enough training data
            entity = None
        else:
            if message.data.get('text_tokens') != None: 
                # print(message.features)               
                # print(message.data)
                tokens = [t.text for t in message.data.get("text_tokens")]
                print("tokens: ", tokens)
                tb = self.preprocessing(tokens)
                print(tb)
                pred = self.clf.prob_classify(tb)
                print(pred.max())

                sentiment_NB = pred.max()
                confidence_NB = pred.prob(sentiment_NB)

                entity_NB = self.convert_to_rasa(sentiment_NB, confidence_NB, "sentiment_NB")
                message.set("entity_NB", [entity_NB], add_to_output=True)

                features = []
                for f in message.features:
                    if f.type == "sentence": 
                        features.append(f.features[0,:])  
                sentiment_SVC = self.clf1.predict(features)
                confidence_SVC = self.clf1.decision_function(features)

                entity_SVC = self.convert_to_rasa(sentiment_SVC, confidence_SVC, "entity_SVC")
                encoded_entity_SVC = json.dumps(entity_SVC, cls=NumpyArrayEncoder)
                message.set("entity_SVC", [encoded_entity_SVC], add_to_output=True)
                

    def persist(self, file_name, model_dir):
        """Persist this model into the passed directory."""
        classifier_file = os.path.join(model_dir, SENTIMENT_MODEL_FILE_NAME)
        utils_io.pickle_dump(classifier_file, self)
        return {"classifier_file": SENTIMENT_MODEL_FILE_NAME}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        file_name = meta.get("classifier_file")
        classifier_file = os.path.join(model_dir, file_name)
        return utils_io.pickle_load(classifier_file)