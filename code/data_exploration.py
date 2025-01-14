import pandas as pd

def data_exploration(data):
    print("\n--- Dataset Overview ---")
    print(f"Shape of the dataset: {data.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(data.head())

    print("\nColumn Information:")
    print(data.info())


    print("\nMissing Values by Column:")
    print(data.isnull().sum())

    # Sum of all missing values
    print(f"\nTotal Missing Values: {data.isnull().sum().sum()}")

    # Check for duplicates
    duplicate_count = data.duplicated().sum()
    print(f"\nNumber of Duplicate Rows: {duplicate_count}")

    # Unique values per column
    print("\nUnique Values per Column:")
    for column in data.columns:
        print(f"{column}: {data[column].nunique()}")

    if 'target' in data.columns:
        print("\nTarget Distribution:")
        print(data['target'].value_counts(normalize=True) * 100)

        print("\nTarget Mapping:")
        print("""
         0 = Stress
         1 = Depression
         2 = Bipolar Disorder
         3 = Personality Disorder
         4 = Anxiety
         """)

    print("\n--- Sample Data ---")
    print(data.head())
    if 'text' in data.columns and 'target' in data.columns:
        print("\nSample Text and Target Columns:")
        print(data[['text', 'target']].head())
