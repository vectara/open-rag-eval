from abc import ABC

class RetrievalMetric(ABC):
    """This class is the base class for all retrieval metrics."""
    pass

class AugmentedGenerationMetric(ABC):
    """This class is the base class for all augmented generation metrics."""
    pass

class PairwiseAnswerSimilarityMetric(ABC):
    """Base class for all pairwise answer similarity metrics."""
    pass
