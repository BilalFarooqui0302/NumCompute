"""Reproducible loop/vectorisation and streaming buffer-size benchmarks."""
from pathlib import Path
import sys, time, json, platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from numcompute_stream import DecisionTreeClassifier


def loop_variance(x):
    mean=0.0
    for v in x: mean += float(v)
    mean /= len(x)
    total=0.0
    for v in x: total += (float(v)-mean)**2
    return total/len(x)

def vector_variance(x): return float(np.var(x))

def median_time(fn,repeats=7):
    fn(); vals=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return float(np.median(vals))

def make_data(n=720,seed=21):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,4))
    signal=1.25*X[:,0]-0.8*X[:,1]+0.55*X[:,2]*X[:,3]
    y=(signal>0).astype(int)
    return X,y

def main():
    rng=np.random.default_rng(42); values=rng.normal(size=120_000)
    t_loop=median_time(lambda:loop_variance(values),repeats=5)
    t_vec=median_time(lambda:vector_variance(values),repeats=5)
    np.testing.assert_allclose(loop_variance(values),vector_variance(values),rtol=1e-10,atol=1e-10)
    X,y=make_data(); chunks=[(X[i:i+40],y[i:i+40]) for i in range(0,len(y),40)]
    results=[]
    for size in (40,80,160,240):
        model=DecisionTreeClassifier(max_depth=5,buffer_size=size,random_state=7)
        correct=total=0; train_times=[]; retrain_times=[]; last_model=None
        for cx,cy in chunks:
            try:
                pred=model.predict(cx); correct += int(np.sum(pred==cy)); total += len(cy)
            except RuntimeError: pass
            before_retrains=model.retrain_count_
            t=time.perf_counter(); model.partial_fit(cx,cy); elapsed=time.perf_counter()-t; train_times.append(elapsed)
            if model.retrain_count_ > before_retrains: retrain_times.append(elapsed)
        # Final holdout score uses the last fitted tree; stream is chronological.
        holdout=np.random.default_rng(999).normal(size=(120,4)); hy=(1.25*holdout[:,0]-.8*holdout[:,1]+.55*holdout[:,2]*holdout[:,3]>0).astype(int)
        holdout_acc=model.score(holdout,hy) if model.root_ is not None else float('nan')
        results.append({"buffer_size":size,"prequential_accuracy":correct/total if total else None,
                        "holdout_accuracy":holdout_acc,"median_chunk_update_ms":float(np.median(train_times)*1000),
                        "median_retrain_event_ms":float(np.median(retrain_times)*1000) if retrain_times else 0.0,
                        "mean_retrain_event_ms":float(np.mean(retrain_times)*1000) if retrain_times else 0.0,
                        "total_retrains":model.retrain_count_,"remaining_buffer":model.buffered_samples_})
    report={"environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
            "vectorisation":{"n":len(values),"loop_median_ms":t_loop*1000,"numpy_median_ms":t_vec*1000,"speedup":t_loop/t_vec},
            "buffer_experiment":results}
    out=ROOT/'benchmark'/'results.json'; out.write_text(json.dumps(report,indent=2))
    md=["# Benchmark results","",f"Environment: Python {report['environment']['python']}, NumPy {report['environment']['numpy']}, {report['environment']['platform']}.","",
        "## Loop versus vectorised variance","",f"- Loop median: {t_loop*1000:.3f} ms",f"- NumPy median: {t_vec*1000:.3f} ms",f"- Speedup: {t_loop/t_vec:.2f}x","- Numerical equivalence checked with `assert_allclose`.","",
        "## Streaming buffer-size experiment","","Synthetic deterministic binary dataset (720 samples, four numeric features), 40-row chunks, depth 5. Accuracy is prequential: each chunk is predicted before it is added to the training buffer.","",
        "| Buffer size | Prequential accuracy | Holdout accuracy | Median update (ms) | Median retrain (ms) | Retrains | Remaining buffer |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results: md.append(f"| {r['buffer_size']} | {r['prequential_accuracy']:.3f} | {r['holdout_accuracy']:.3f} | {r['median_chunk_update_ms']:.3f} | {r['median_retrain_event_ms']:.3f} | {r['total_retrains']} | {r['remaining_buffer']} |")
    md += ["","Timings are machine-dependent and are descriptive rather than universal. The final holdout set is independently generated and uses the same synthetic decision boundary."]
    (ROOT/'benchmark'/'results.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))
if __name__=='__main__': main()
