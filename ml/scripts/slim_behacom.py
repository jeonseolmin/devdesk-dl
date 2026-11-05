"""BEHACOM zip → behacom_slim.csv, dropping per-key/digraph columns (keystroke biometrics, out of DevDesk scope).

Usage: python -I ml/scripts/slim_behacom.py ml/data/raw/behacom.zip ml/data/raw/behacom/behacom_slim.csv
The zip keeps every column; re-run with a different DROP to get them back.
"""
import csv
import io
import sys
import zipfile

DROP = ("digraph_", "keystrokes_key_", "press_release_average_")


def main(zip_path, out_path):
    csv.field_size_limit(2**31 - 1)
    total = 0
    # surrogateescape: some app names are not valid UTF-8 (e.g. Latin-1 "Mamá"); keep their bytes unchanged
    with zipfile.ZipFile(zip_path) as z, open(out_path, "w", newline="", encoding="utf-8", errors="surrogateescape") as out:
        w = csv.writer(out)
        names = sorted(n for n in z.namelist() if n.endswith("_BEHACOM.csv"))
        assert len(names) == 12, names
        header_out = None
        for name in names:
            with z.open(name) as raw:
                r = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="surrogateescape", newline=""))
                header = next(r)
                keep = [i for i, h in enumerate(header) if not h.startswith(DROP)]
                cols = [header[i] for i in keep]
                if header_out is None:
                    header_out = cols
                    w.writerow(cols)
                assert cols == header_out, f"{name}: header differs"
                n = 0
                for row in r:
                    assert len(row) == len(header), f"{name} row {n}: {len(row)} != {len(header)}"
                    w.writerow([row[i] for i in keep])  # values copied as-is, no float round-trip
                    n += 1
            print(f"{name}: {n} rows")
            total += n
    print(f"total {total} rows, {len(header_out)} columns")


if __name__ == "__main__":
    main(*sys.argv[1:3])
