import json
import math
from pathlib import Path


def convert_size(size_bytes: int):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def make_human_readable(fname: str):
    data = json.loads(Path(fname).read_text(encoding="utf-8"))

    for key, value in data.items():
        if "_mem_" in key:
            if isinstance(value, int):
                data[key] = convert_size(value)

    with open(fname, "w", encoding="utf-8") as fhout:
        json.dump(data, fhout, ensure_ascii=False, indent=4)
