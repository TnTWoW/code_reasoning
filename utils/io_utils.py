import json
import pickle

def read_robustfill_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 将 JSON 对象中的值转换为列表
            data_list = [value for key, value in sorted(data.items(), key=lambda item: int(item[0]))]
            return data_list
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode error: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

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
