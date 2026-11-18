"""Prequential streaming trainer with per-chunk logs and memory estimates."""
import numpy as np
from .metrics import StreamingClassificationMetrics

class StreamTrainer:
    def __init__(self, model, pipeline=None, rolling_window=None):
        self.model=model; self.pipeline=pipeline
        self.metrics=StreamingClassificationMetrics(window_size=rolling_window)
        self.logs=[]; self.chunk_index=0
    @property
    def estimator(self): return self.pipeline if self.pipeline is not None else self.model
    def score_chunk(self,X,y):
        pred=self.estimator.predict(X)
        self.metrics.update(y,pred)
        return pred
    def fit_chunk(self,X,y):
        X=np.asarray(X); y=np.asarray(y)
        if X.ndim!=2 or X.shape[0]!=len(y) or len(y)==0: raise ValueError("X must be 2D and match non-empty y")
        before=None
        try: before=self.estimator.predict(X)
        except RuntimeError: pass
        prequential=None if before is None else float(np.mean(before==y))
        # Evaluate before learning from this chunk to avoid optimistic leakage.
        if before is not None: self.metrics.update(y,before)
        if self.pipeline is not None: self.pipeline.partial_fit(X,y)
        else: self.model.partial_fit(X,y)
        after=None
        try: after=self.estimator.predict(X)
        except RuntimeError: pass
        result=self.metrics.result() if self.metrics._y else {"accuracy":float("nan"),"precision":float("nan"),"recall":float("nan"),"f1":float("nan")}
        self.chunk_index+=1
        model=self.model
        log={"chunk":self.chunk_index,"n_samples":len(y),"prequential_accuracy":prequential,
             "post_update_accuracy":None if after is None else float(np.mean(after==y)),
             "cumulative_accuracy":result["accuracy"],"precision":result["precision"],"recall":result["recall"],"f1":result["f1"],
             "retrain_count":getattr(model,"retrain_count_",None),"buffered_samples":getattr(model,"buffered_samples_",None),
             "memory_bytes":int(sum(a.nbytes for a in getattr(model,"_buffer_X",[]))+sum(a.nbytes for a in getattr(model,"_buffer_y",[])))}
        self.logs.append(log); return log
    def score(self,X,y): return float(np.mean(self.estimator.predict(X)==np.asarray(y)))
    def reset(self): self.metrics.reset(); self.logs.clear(); self.chunk_index=0; return self
