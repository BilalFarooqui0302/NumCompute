"""CSV helpers for the demonstration, implemented with NumPy and stdlib."""
from pathlib import Path
import numpy as np

def read_numeric_csv(path, delimiter=",", skip_header=1, usecols=None):
    return np.genfromtxt(Path(path),delimiter=delimiter,skip_header=skip_header,usecols=usecols,dtype=float,ndmin=2,encoding="utf-8")

def iter_numeric_csv_chunks(path,chunksize,delimiter=",",skip_header=1,usecols=None):
    if chunksize<1: raise ValueError("chunksize must be positive")
    with open(path,"r",encoding="utf-8",newline="") as f:
        for _ in range(skip_header): next(f,None)
        while True:
            lines=[]
            for _ in range(chunksize):
                line=f.readline()
                if not line: break
                lines.append(line)
            if not lines: break
            a=np.genfromtxt(lines,delimiter=delimiter,usecols=usecols,dtype=float,ndmin=2)
            if a.size: yield a
