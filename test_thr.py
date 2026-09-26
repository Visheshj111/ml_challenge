import time
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

s1 = ['A long string to compare with rapidfuzz'] * 1_000_000
s2 = ['Another string that is somewhat different'] * 1_000_000

t0 = time.time()
[fuzz.ratio(a,b) for a,b in zip(s1,s2)]
t_seq = time.time() - t0
print(f'Seq: {t_seq:.2f}s')

def chunk_fuzz(chunk):
    return [fuzz.ratio(a,b) for a,b in chunk]

chunks = [list(zip(s1[i:i+10000], s2[i:i+10000])) for i in range(0, 1_000_000, 10000)]

t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    list(ex.map(chunk_fuzz, chunks))
t_thr = time.time() - t0
print(f'Thread: {t_thr:.2f}s')
