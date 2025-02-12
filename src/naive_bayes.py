import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features[0].shape[0] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples

        total_labels = labels.numel()  # Number of total samples
        each_label_count = labels.to(torch.int64).bincount()  # Number of samples of each class
        class_priors: Dict[int, torch.Tensor] = {labels[i].item(): torch.Tensor((each_label_count[i]/total_labels)) for i in range(each_label_count.numel())}

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing

        class_word_counts: Dict[int, torch.Tensor] = dict()

        for label in labels.unique():
            # Indexes that have the specific label
            label_indexes = labels == label
            # To take all the features of that label
            label_features = features[label_indexes]

            # Sum all de elements of the columns to know how many words of each type we have
            label_word_counts = label_features.sum(dim=0)

            # Sum all the elements to know how many words we have in the label
            label_total_words = label_features.sum()

            # Compute conditional probability and add to dict
            label_cond_prob = (label_word_counts+delta)/(label_total_words + self.vocab_size*delta)
            class_word_counts[label.item()] = label_cond_prob

        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words

        # Words that are in the document we want to classify
        present_words = feature != 0

        # Total classes
        classes = self.class_priors.keys()

        log_posteriors: torch.Tensor = torch.zeros(len(classes))

        for i in range(len(classes)):
            c = list(classes)[i]

            # Prior log(p(c))
            p_c = self.class_priors[c]  
            log_p_c = torch.log(p_c)


            cond_prob = self.conditional_probabilities[c]
            log_cond_prob = torch.log(cond_prob)

            present_cond_prob = log_cond_prob[present_words]  # Take into account only the ones that are present in the document
            likelihood = present_cond_prob.sum()  # sum(log(p(w1|c)),...,log(p(wn|c)))

            # p(c|w1,..,wn) = sum(log(p(w1|c)),...,log(p(wn|c))) + log(p(c))
            log_posterior = likelihood + log_p_c

            log_posteriors[i] = log_posterior

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 

        posteriors = self.estimate_class_posteriors(feature)

        # Take the one with highest argument
        pred: int = torch.argmax(posteriors).item()

        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        posteriors = self.estimate_class_posteriors(feature)

        # Probability distribution
        probs: torch.Tensor = torch.softmax(posteriors, dim=-1)

        return probs
