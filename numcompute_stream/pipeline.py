"""Small streaming pipeline connecting incremental transformers to a model."""
import numpy as np

class Pipeline:
    def __init__(self, steps):
        if not steps or len(steps)<1: raise ValueError("steps must not be empty")
        self.steps=list(steps)
        names=[n for n,_ in self.steps]
        if len(set(names))!=len(names): raise ValueError("step names must be unique")
        self.transformers=self.steps[:-1]; self.model=self.steps[-1][1]
        if not hasattr(self.model,"partial_fit"): raise TypeError("final step must implement partial_fit")
        self._input_features=None; self._output_features=None
    def _transform(self,X):
        a=np.asarray(X)
        for _,transformer in self.transformers: a=transformer.transform(a)
        return a
    def partial_fit(self,X_chunk,y_chunk):
        a=np.asarray(X_chunk)
        if a.ndim!=2 or a.shape[0]==0: raise ValueError("X_chunk must be non-empty and 2D")
        if self._input_features is not None and a.shape[1]!=self._input_features: raise ValueError("Input feature count changed")
        self._input_features=a.shape[1]
        # Update each transformer on the representation it receives.
        current=a
        old_dims=[]
        for _,tr in self.transformers:
            before=getattr(tr,"n_features_out_",None)
            if hasattr(tr,"partial_fit"): tr.partial_fit(current)
            after=getattr(tr,"n_features_out_",None)
            old_dims.append((before,after))
            current=tr.transform(current)
        if self._output_features is not None and current.shape[1]!=self._output_features:
            # A growing encoder changes the model input schema. Reset model and its buffer
            # rather than mixing incompatible feature spaces.
            if hasattr(self.model,"reset_stream"): self.model.reset_stream()
            else: raise ValueError("Transformer output shape changed; model cannot be reset")
        self._output_features=current.shape[1]
        self.model.partial_fit(current,y_chunk)
        return self
    def predict(self,X): return self.model.predict(self._transform(X))
    def score(self,X,y): return float(np.mean(self.predict(X)==np.asarray(y)))
