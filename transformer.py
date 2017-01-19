import pandas as pd

class Transformer:
    def __init__(self, untouched=['id', 'timestamp', 'y']):
        '''
        Inputs:
        - untouched: A list of column names to ignore when transforming the dataframes
        '''
        self.untouched = untouched
    def fit(self, df):
        '''
        Inputs:
        - df: A dataframe to extract metadata (median, mean and std, of each column)
        '''
        self.median = df.median()
        self.mean = df.mean()
        self.std = df.std()
    def transform(self, df):
        '''
        Given a dataframe,
        it fills the missing attributes with the median of each column,
        limits the values of each column to between 5 STD from the mean,
        and normalizes the features
        '''
        # Fill the missing attributes with the median of each column
        df.fillna(self.median, inplace=True)
        # Limit the values of each column to between 5 STD from the mean
        col_names = [col for col in df.columns if col not in self.untouched]
        for cn in col_names:
            lower = self.mean[cn] - 5 * self.std[cn]
            upper = self.mean[cn] + 5 * self.std[cn]
            df.loc[:, cn] = df.loc[:, cn].clip(lower=lower, upper=upper)
        # Normalize the features
        for cn in col_names:
            df.loc[:, cn] = (df.loc[:, cn] - self.mean[cn]) / self.std[cn]
        return df