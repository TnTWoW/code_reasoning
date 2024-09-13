import json
import pickle


def read_jsonl(filename, n=None):
    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append(json.loads(line.rstrip()))
    if n is not None:
        lines = lines[:n]
    return lines


def write_jsonl(lines, filename, indent=None):
    with open(filename, "w") as f:
        for line in lines:
            json_record = json.dumps(line, indent=indent)
            f.write(json_record + "\n")


def read_txt(filename):
    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append((line.rstrip()))
    return lines


def write_txt(lines, filename):
    with open(filename, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_tsv(filename, separator="\t"):
    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append((line.rstrip().split(separator)))
    return lines


def write_tsv(lines, filename, separator="\t"):
    with open(filename, "w") as f:
        for line in lines:
            line = separator.join(line)
            f.write(f"{line}\n")


def write_json(json_dict, filename, indent=2):
    with open(filename, "w") as f:
        json.dump(json_dict, f, indent=indent)


def read_json(filename):
    with open(filename, "r") as f:
        json_dict = json.load(f)
    return json_dict


def read_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def write_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
