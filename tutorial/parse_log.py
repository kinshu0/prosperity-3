import json
import pandas as pd
import re

def parse_log(log_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(log_file_path, 'r', encoding='utf-8') as file:
        log_data = file.read()
    
    # Find indices of sections
    sandbox_index = log_data.find("Sandbox logs:")
    activities_index = log_data.find("Activities log:")
    trade_index = log_data.find("Trade History:")
    
    # Extract raw strings
    sandbox_logs_raw = log_data[sandbox_index + len("Sandbox logs:"):activities_index].strip()
    activities_log_raw = log_data[activities_index + len("Activities log:"):trade_index].strip()
    trade_history_raw = log_data[trade_index + len("Trade History:"):].strip()
    
    # Parse Sandbox logs (JSON objects)
    sandbox_logs = [json.loads(obj) for obj in re.findall(r'\{.*?\}', sandbox_logs_raw, re.DOTALL)]
    sandbox_df = pd.DataFrame(sandbox_logs)
    
    # Parse Trade History (JSON list)
    trade_history = json.loads(trade_history_raw) if trade_history_raw.startswith('[') else []
    trade_history_df = pd.DataFrame(trade_history)
    
    # Read Activities log as CSV
    from io import StringIO
    market_data = pd.read_csv(StringIO(activities_log_raw), delimiter=';')
    
    return sandbox_df, market_data, trade_history_df

if __name__ == "__main__":
    # Example usage
    log_file_path = "logs/empty_submission.log"  # Change this to the actual log file path
    sandbox_logs, market_data, trade_history = parse_log(log_file_path)