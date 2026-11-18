"""Reusable matplotlib plots for streaming model diagnostics."""
import numpy as np

def _finish(fig, save_path=None, show=True):
    if save_path: fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig

def plot_metric_over_time(metric_values, title="Metric over streaming chunks", ylabel="Metric", *, xlabel="Chunk", save_path=None, show=True):
    import matplotlib.pyplot as plt
    values=np.asarray(metric_values,dtype=float).reshape(-1)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(1,len(values)+1),values,marker="o")
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(True,alpha=.25); fig.tight_layout()
    return _finish(fig,save_path,show)

def compare_models(metric1, metric2, labels=("Model 1","Model 2"), *, title="Streaming model comparison", ylabel="Accuracy", save_path=None, show=True):
    import matplotlib.pyplot as plt
    a=np.asarray(metric1,dtype=float).reshape(-1); b=np.asarray(metric2,dtype=float).reshape(-1)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(1,len(a)+1),a,marker="o",label=labels[0]); ax.plot(np.arange(1,len(b)+1),b,marker="s",label=labels[1])
    ax.set(title=title,xlabel="Chunk",ylabel=ylabel); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
    return _finish(fig,save_path,show)

def plot_predictions_vs_ground_truth(y_true,y_pred, *, title="Predictions vs ground truth", save_path=None, show=True):
    import matplotlib.pyplot as plt
    yt=np.asarray(y_true).reshape(-1); yp=np.asarray(y_pred).reshape(-1)
    if yt.shape!=yp.shape: raise ValueError("y_true and y_pred must have matching shapes")
    fig,ax=plt.subplots(figsize=(8,3.8)); x=np.arange(len(yt)); ax.scatter(x,yt,marker="o",label="Ground truth"); ax.scatter(x,yp,marker="x",label="Prediction")
    ax.set(title=title,xlabel="Sample in latest chunk",ylabel="Class label"); ax.legend(); ax.grid(True,alpha=.2); fig.tight_layout()
    return _finish(fig,save_path,show)
