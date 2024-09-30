from datetime import datetime

import pandas as pd
from datasets import load_dataset
import json

pd.set_option('display.max_columns', None)
data_files = {"test": "./execution-v2/data/test-00000-of-00001.parquet"}
test_sets = load_dataset("parquet", data_files=data_files)["test"]

time_line = datetime(2023, 10, 1, 0, 0)
filtered_data = test_sets.filter(lambda x: x["contest_date"] > time_line)
# def convert_date_to_str(examples):
#     # 假设 contest_date 是 datetime 类型，转换为字符串格式
#     examples['contest_date'] = [date.strftime('%Y-%m-%d') if isinstance(date, datetime.datetime) else date for date in examples['contest_date']]
#     return examples
#
# filtered_data = filtered_data.map(convert_date_to_str, batched=True)

data_list = filtered_data.to_list()
for record in data_list:
    if 'contest_date' in record and isinstance(record['contest_date'], datetime):
        record['contest_date'] = record['contest_date'].strftime('%Y-%m-%d')

with open('livecodebench.jsonl', 'w') as file:
    for record in data_list:
        file.write(json.dumps(record) + '\n')


def fn(grid):
    # Step 1: Identify the 2x2 pattern from
    # the top-left corner of the input grid
    pattern = [[grid[0][0], grid[0][1]],
               [grid[1][0], grid[1][1]]]
    # Step 2: Extend this pattern to fill a 5x5 grid
    result = []
    for i in range(5):
        row = []
        for j in range(5):
            # Use modulo operation to repeat the 2x2 pattern
            row.append(pattern[i % 2][j % 2])
        result.append(row)
    return result

def fn(grid):
    # Find the first row from the bottom that contains a '1'
    first_row_with_one = None
    for i in range(len(grid) - 1, -1, -1):
        if 1 in grid[i]:
            first_row_with_one = i
            break
            # If no row contains a '1', return the grid as is
    if first_row_with_one is None:
        return grid
    # Copy the identified row upwards
    for i in range(first_row_with_one - 1, -1, -1):
        grid[i] = grid[first_row_with_one][:]
    return grid

