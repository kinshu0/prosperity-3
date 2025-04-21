import pandas as pd
from io import StringIO
import re
import orjson

def read_submission_log(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(filename, 'r') as f:
        log_content = f.read()

    sandbox, market = log_content.split("Sandbox logs:")[1].split("Activities log:")
    market, trade_history = market.split("Trade History:")

    market = pd.read_csv(StringIO(market.strip()), sep=';')
    trade_history = orjson.loads(trade_history.strip())
    trade_history = pd.DataFrame(trade_history)
    sandbox = [orjson.loads(s) for s in re.findall(r'\{.*?\}', sandbox, re.DOTALL)]
    sandbox = pd.DataFrame(sandbox)
    return market, trade_history, sandbox