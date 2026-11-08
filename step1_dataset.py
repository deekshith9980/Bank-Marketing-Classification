import pandas as pd
import requests
import zipfile

from io import BytesIO


def load_dataset():

    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    outer_zip = zipfile.ZipFile(BytesIO(response.content))

    inner_zip_data = outer_zip.read("bank.zip")

    inner_zip = zipfile.ZipFile(BytesIO(inner_zip_data))

    with inner_zip.open("bank-full.csv") as file:
        df = pd.read_csv(file, sep=";")

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df["y"].value_counts())

    print("\nTarget Distribution (%):")
    print(df["y"].value_counts(normalize=True) * 100)