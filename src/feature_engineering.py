def add_features(df):
    df = df.copy()
    df['description_length'] = df['Details'].astype(str).apply(len)
    df['title_word_count'] = df['Title'].astype(str).apply(lambda x: len(x.split()))
    df['details_word_count'] = df['Details'].astype(str).apply(lambda x: len(x.split()))
    df.drop(columns=['Title', 'Details'], inplace=True)
    return df
