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

SENTIMENT_MODEL_FILE_NAME = "sentiment_classifier.pkl"


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
        for t in training_data:
            if t.data.get('text_tokens') == None:
                continue
            tokens.append(map(lambda x: x.text, t.data.get('text_tokens')))
        processed_tokens = [self.preprocessing(t) for t in tokens]
        labeled_data = [(t, x) for t,x in zip(processed_tokens, labels)]
        self.clf = NaiveBayesClassifier.train(labeled_data)


    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

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
                print(message.data.get('text_tokens'))
                tokens = [t.text for t in message.data.get("text_tokens")]
                print("tokens: ", tokens)
                tb = self.preprocessing(tokens)
                print(tb)
                pred = self.clf.prob_classify(tb)
                print(pred.max())

                sentiment = pred.max()
                confidence = pred.prob(sentiment)

                entity = self.convert_to_rasa(sentiment, confidence)

                message.set("entities", [entity], add_to_output=True)

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