import pandas as pd
import numpy as np


df = pd.read_csv("messy_sales_data.csv") 
print("Initial Data Info:")
print(df.info()) [cite: 8]


df = df.drop_duplicates() [cite: 8, 28]


cols_to_drop = ["notes", "temp_id"] 
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns]) [cite: 28]


if "revenue" in df.columns:
    df["revenue"] = df["revenue"].fillna(df["revenue"].median()) [cite: 30]

if "customer_id" in df.columns:
    df = df.dropna(subset=["customer_id"]) [cite: 30]


if "sale_date" in df.columns:
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce") [cite: 32]

if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace("$", ""), errors="coerce") [cite: 33]


if "product" in df.columns:
    df["product"] = df["product"].str.lower().str.strip() [cite: 35]

if "region" in df.columns:
    region_map = {"west": "Western", "SOUTH": "Southern"}
    df["region"] = df["region"].replace(region_map) [cite: 36]


assert df.duplicated().sum() == 0, "Duplicates still exist!" [cite: 38]
print("Cleaning Complete. Final Shape:", df.shape)

df.to_csv("cleaned_sales_data.csv", index=False) [cite: 37]