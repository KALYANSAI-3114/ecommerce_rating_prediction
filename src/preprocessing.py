def preprocess_data(df):
    df = df.copy()
    df.ffill(inplace=True)
    return df
