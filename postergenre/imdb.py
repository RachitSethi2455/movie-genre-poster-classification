"""IMDb genres for the posters (files are named by IMDb title ID, e.g. ``tt2943178.jpg``).

Uses IMDb's non-commercial dataset ``title.basics.tsv.gz``. It is downloaded on demand and never committed;
only derived statistics are. Information courtesy of IMDb (https://www.imdb.com). Used with permission.
"""
import urllib.request
from pathlib import Path

import pandas as pd

IMDB_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"


def download_title_basics(path="data/imdb/title.basics.tsv.gz"):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMDB_URL, path)  # about 230 MB
    return path


def imdb_genres(title_ids, path=None, chunksize=500_000):
    """{title_id: set of IMDb genres} for the given IDs. IDs missing from IMDb are left out.

    The file has about 12M rows, so it is streamed in chunks and filtered, which keeps memory low.
    """
    path = download_title_basics() if path is None else Path(path)
    wanted, found = set(title_ids), {}
    for chunk in pd.read_csv(path, sep="\t", usecols=["tconst", "genres"], dtype=str, na_values=["\\N"],
                             chunksize=chunksize, quoting=3):  # quoting=3: titles contain stray quotes
        hits = chunk[chunk["tconst"].isin(wanted)].dropna(subset=["genres"])
        found.update({t: set(g.split(",")) for t, g in zip(hits["tconst"], hits["genres"])})
    return found
