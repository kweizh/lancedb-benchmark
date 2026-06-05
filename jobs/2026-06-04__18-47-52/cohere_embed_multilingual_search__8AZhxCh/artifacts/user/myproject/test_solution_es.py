import os
from solution import cross_lingual_search

os.environ["ZEALT_RUN_ID"] = "test1234"
res = cross_lingual_search("¿Quién pintó la Mona Lisa?", k=3)
for r in res:
    print(r)
