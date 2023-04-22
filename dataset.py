""" Load and prepare Cornell Movie Dialog dataset , a famous dataset for training a chatbot model."""

import os
import re
import dataclasses
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from typing import Union, Tuple

tokenizer_type = tfds.deprecated.text.subword_text_encoder.SubwordTextEncoder


class HyperParameters(object):
    """This class helps to hold all the required hyper-parameters
    in one place. we can easily add new parameters to it."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def remove_contractions(sentence: str) -> str:
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    return sentence


def preprocess_sentence(sentence: str) -> str:
    """Clean the input sentence by adding whitespace between punctuations and characters and
    normalize the sentence."""
    sentence = sentence.lower().strip()

    # create a space between a word and the punctuation following it.
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # removing contractions
    sentence = remove_contractions(sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()

    return sentence


def load_conversations(lines_filename: str, conversations_filename: str, max_sample: Union[int, None] = None):
    """Loads the conversations from cornel dialog dataset. It works based on structure of this
    specific dataset and can not be used for other datasets."""
    # dictionary of line id to text
    id2line = {}
    with open(lines_filename, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    # Load conversations
    questions, answers = [], []
    with open(conversations_filename, "r") as file:
        lines = file.readlines()
    for line in tqdm(lines):
        parts = line.replace("\n", "").split(" +++$+++ ")
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            questions.append(preprocess_sentence(id2line[conversation[i]]))
            answers.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if max_sample and max_sample >= len(questions):
                return questions, answers
    return questions, answers

