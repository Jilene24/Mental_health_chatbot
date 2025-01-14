def data_cleaning(data):
    duplicates = data.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")
    if duplicates.sum() > 0:
        print("Duplicate rows:")
        print(data[duplicates])

    # Remove duplicates
    data = data.drop_duplicates()
    print(f"Shape after removing duplicates: {data.shape}")

    print("\n--- Checking for Missing Values ---")
    missing_values = data.isnull().sum()
    print(f"Missing values per column:\n{missing_values}")

    # Remove rows with missing values
    data = data.dropna()
    print(f"Shape after removing rows with missing values: {data.shape}")

    return data
