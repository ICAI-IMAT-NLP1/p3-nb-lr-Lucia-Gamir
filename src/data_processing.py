from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    
    with open(infile, "r", encoding="utf-8") as txt:
        lines = txt.readlines()

    # Split -> (tokenize(sentence), label)
    separated_lines = [(tokenize(line.split("\t")[0]), line.split("\t")[-1]) for line in lines] 

    # Make the objects
    examples: List[SentimentExample] = [SentimentExample(sep_line[0], int(sep_line[1])) for sep_line in separated_lines]  

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    
    # Put all the words in a list
    total_words = []

    for example in examples:
        total_words += example.words

    # Select unique ones
    total_unique_words = set(total_words)
    
    # Create dictionary
    vocab: Dict[str, int] = {word:index for word, index in zip(total_unique_words, range(len(total_unique_words)))}

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full

    bow: torch.Tensor = torch.Tensor([0 for i in range(len(vocab))])

    if binary:
        for word in text:
            if word in vocab:
                index = vocab[word]
                bow[index] = 1
    else:
        for word in text:
            if word in vocab:
                index = vocab[word]
                bow[index] += 1

    return bow
