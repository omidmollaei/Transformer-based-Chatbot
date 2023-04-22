""" Load and prepare Cornell Movie Dialog dataset , a famous dataset for training a chatbot model."""

import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Union, Tuple, List, NewType, Any

tokenizer_type = tfds.deprecated.text.subword_text_encoder.SubwordTextEncoder
DataClassType = NewType("DataClassType", Any)


@dataclass
class DatasetHp(object):
    """This class helps to hold all the required hyper-parameters in one place.
     we can easily add new parameters to it."""
    max_length: int
    start_token: List[int] = field(default_factory=[0])
    end_token: List[int] = field(default_factory=[1])
    vocab_size: int = 2**13
    batch_size: int = 32
    max_sample: Union[int, None] = None


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


def tokenize_and_filter(hparams: DataClassType, tokenizer: tokenizer_type, questions: list,
                        answers: list) -> Tuple[list, list]:
    """Converts the clean text into tokens using the given tokenizer."""
    tokenized_questions, tokenized_answers = [], []
    for (question, answer) in tqdm(zip(questions, answers)):
        # tokenize sentence
        sentence1 = hparams.start_token + tokenizer.encode(question) + hparams.end_token
        sentence2 = hparams.start_token + tokenizer.encode(answer) + hparams.end_token

        # check tokenize sentence length
        if (len(sentence1) < hparams.max_length) and (len(sentence2) <= hparams.max_length):
            tokenized_questions.append(sentence1)
            tokenized_answers.append(sentence2)

    # pad tokenized sentences
    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=hparams.max_length, padding="post")
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=hparams.max_length, padding="post")

    return tokenized_questions, tokenized_answers


def get_cornell_dataset(hparams: DataClassType) -> Tuple[tf.data.Dataset, tokenizer_type]:

    # download corpus
    path_to_zip = tf.keras.utils.get_file(
        "cornell_movie_dialogs.zip",
        origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        extract=True,
    )

    path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    # get movie_lines.txt and movie_conversations.txt
    lines_filename = os.path.join(path_to_dataset, "movie_lines.txt")
    conversations_filename = os.path.join(path_to_dataset, "movie_conversations.txt")

    print("loading conversations ... ")
    questions, answers = load_conversations(hparams, lines_filename, conversations_filename)

    print("initializing tokenizer ...")
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=hparams.vocab_size)
    tokenizer.save_to_file(os.path.join("./", "transformer", "tokenizer"))  # save tokenizer

    hparams.start_token = [tokenizer.vocab_size]
    hparams.end_token = [tokenizer.vocab_size + 1]
    hparams.vocab_size = tokenizer.vocab_size + 2

    print("tokenization ... ")
    questions, answers = tokenize_and_filter(hparams, tokenizer, questions, answers)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"inputs": questions, "dec_inputs": answers[:, :-1]}, answers[:, 1:])
    )

    dataset = dataset.cache()
    dataset = dataset.shuffle(len(questions))
    dataset = dataset.batch(hparams.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, tokenizer
