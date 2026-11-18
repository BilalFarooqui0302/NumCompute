"""NumCompute streaming decision-tree toolkit (NumPy + matplotlib only)."""
from .tree import DecisionTreeClassifier
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .stats import StreamingStats
from .metrics import StreamingClassificationMetrics
from .pipeline import Pipeline
from .stream import StreamTrainer
__all__=["DecisionTreeClassifier","StandardScaler","Imputer","OneHotEncoder","StreamingStats","StreamingClassificationMetrics","Pipeline","StreamTrainer"]
