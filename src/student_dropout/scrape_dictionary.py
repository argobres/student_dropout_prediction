"""Optional BeautifulSoup dictionary refresh; network required, not part of training."""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
import csv
import json

import requests
from bs4 import BeautifulSoup

url = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

response = requests.get(url, timeout=30)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

for script in soup.select('script[type="application/json"][data-url]'):
    endpoints = script["data-url"].split("?")[0].split("/")[-1].split(",")
    if "variables.findTableByDatasetId" in endpoints:
        index = endpoints.index("variables.findTableByDatasetId")
        payload = json.loads(script.string)
        results = json.loads(payload["body"])
        table = results[index]["result"]["data"]["json"]
        break
else:
    raise RuntimeError("Variables table not found")

with open(args.output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(table["headers"])
    writer.writerows(table["data"])

print(f"Saved {len(table['data'])} variables to variables_table.csv")
