import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_quality.lotus_llm_adapter import LotusLLM

if __name__ == "__main__":
    input_data = json.load(sys.stdin)
    raw_tables = input_data["tables"]  # List of (name, csv_string)

    dfs = []
    from io import StringIO
    for name, table_csv in raw_tables:
        df = pd.read_csv(StringIO(table_csv))
        dfs.append((name, df))

    lotus = LotusLLM(dfs, model=input_data["model"], mode=input_data["mode"])
    response = lotus.run_lotus(input_data["question"])
    print(response)
