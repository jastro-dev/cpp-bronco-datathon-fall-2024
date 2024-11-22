import pandas as pd

df = pd.read_csv('data/flights.csv')
df['to'].value_counts()

def summarize_dataframe(df: pd.DataFrame, df_name: str='DataFrame'):
    """
    Summarize the DataFrame by displaying its shape, missing values,
    data types, and duplicate values.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
        df_name (str): A name for the DataFrame, used in output messages.
    """
    print(f"Summary for {df_name}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nDescriptive Statistics:\n{df.describe(include='all')}")
    print(f"\nUnique Values:\n{df.nunique()}")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nDuplicate Rows: {duplicate_count}")
        print(f"Examples of Duplicate Rows:\n{df[df.duplicated(keep=False)].head()}")
    else:
        print("\nNo Duplicate Rows Found.")

summarize_dataframe(df, 'Travel Dataframe:')

for col in df.columns:
    print(df[col].value_counts())