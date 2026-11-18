"""Cumulative and rolling streaming classification metrics."""
from collections import deque
import numpy as np

class StreamingClassificationMetrics:
    def __init__(self, labels=None, window_size=None):
        if window_size is not None and (not isinstance(window_size,int) or window_size<1): raise ValueError("window_size must be positive or None")
        self.labels = None if labels is None else np.asarray(labels)
        self.window_size=window_size; self.reset()
    def reset(self):
        self._y=[]; self._pred=[]; self._scores=[]; self._has_scores=False
        self._window=deque(maxlen=self.window_size) if self.window_size else None
        self.confusion_matrix_=None
        return self
    def update(self,y_true_chunk,y_pred_chunk,y_score=None):
        yt=np.asarray(y_true_chunk); yp=np.asarray(y_pred_chunk)
        if yt.ndim!=1 or yp.ndim!=1 or yt.shape!=yp.shape or yt.size==0: raise ValueError("y_true and y_pred must be non-empty 1D arrays of equal shape")
        if y_score is not None:
            ys=np.asarray(y_score,dtype=float)
            if ys.ndim!=1 or ys.shape!=yt.shape or np.any(~np.isfinite(ys)): raise ValueError("y_score must be finite 1D and match y_true")
            self._has_scores=True
        else: ys=None
        self._y.append(yt.copy()); self._pred.append(yp.copy())
        self._scores.append(np.full(yt.shape, np.nan) if ys is None else ys.copy())
        if self._window is not None:
            for i in range(len(yt)): self._window.append((yt[i],yp[i],None if ys is None else ys[i]))
        labels=np.unique(np.concatenate((self.labels if self.labels is not None else np.array([],dtype=yt.dtype),yt,yp)))
        self.labels=labels
        yi=np.searchsorted(labels,yt); pi=np.searchsorted(labels,yp)
        cm=np.zeros((len(labels),len(labels)),dtype=np.int64); np.add.at(cm,(yi,pi),1)
        if self.confusion_matrix_ is not None and np.array_equal(labels,self._labels_cm): self.confusion_matrix_ += cm
        else:
            if self.confusion_matrix_ is None: self.confusion_matrix_=cm
            else:
                old=self.confusion_matrix_; oldlabels=self._labels_cm
                expanded=np.zeros_like(cm); idx=np.searchsorted(labels,oldlabels); expanded[np.ix_(idx,idx)]=old; self.confusion_matrix_=expanded+cm
        self._labels_cm=labels.copy()
        return self
    def _arrays(self, rolling=False):
        if rolling:
            if self._window is None: raise ValueError("window_size was not configured")
            if not self._window: raise RuntimeError("No metric observations")
            z=list(self._window); return np.asarray([v[0] for v in z]),np.asarray([v[1] for v in z])
        if not self._y: raise RuntimeError("No metric observations")
        return np.concatenate(self._y),np.concatenate(self._pred)
    def result(self, rolling=False):
        yt,yp=self._arrays(rolling); labels=np.unique(np.concatenate((yt,yp)))
        cm=np.zeros((len(labels),len(labels)),dtype=float); np.add.at(cm,(np.searchsorted(labels,yt),np.searchsorted(labels,yp)),1)
        tp=np.diag(cm); fp=cm.sum(axis=0)-tp; fn=cm.sum(axis=1)-tp
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        recall=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
        f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)>0)
        out={"accuracy":float(np.mean(yt==yp)),"precision":float(np.mean(precision)),"recall":float(np.mean(recall)),"f1":float(np.mean(f1)),"confusion_matrix":cm.astype(int),"labels":labels}
        if not rolling and self._has_scores and len(np.unique(yt))==2:
            scores=np.concatenate(self._scores); pos=labels[-1]
            out["roc_auc"]=self._auc(yt==pos,scores) if np.all(np.isfinite(scores)) else float("nan")
        return out
    @staticmethod
    def _auc(y,s):
        n1=int(y.sum()); n0=len(y)-n1
        if n1==0 or n0==0: return float("nan")
        order=np.argsort(s,kind="mergesort"); ranks=np.empty(len(s),float); sorted_s=s[order]
        i=0
        while i<len(s):
            j=i+1
            while j<len(s) and sorted_s[j]==sorted_s[i]: j+=1
            ranks[order[i:j]]=(i+1+j)/2.0; i=j
        return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))
