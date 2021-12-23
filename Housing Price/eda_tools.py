import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

import fuzzywuzzy
from fuzzywuzzy import process

def check_matches(df, limit=5, score = 70):
    '''
     Requirement:
     Must be in DataFrame
     Use for categorical data

     Data entry error and mispelling can result in labels that are spelled differently but meant the same thing
     1) check each categorical col for similarity among labels in each column
         - a score will be given for the level of similarity to each other
     2) Will replace all np.nan values in categorical cols with null cells as 'NA' as fuzzywuzzy cannot work with np.nan

     return 2 dict, all_matches and highlight
     1) all_matches which showcase all the columns and score for each label similarity
     2) highlight which only showcase columns which as labels that are highly similar to each other
    '''
    assert(isinstance(df, pd.DataFrame)),'First argument: input must be pandas DataFrame'
    duplicate = df.duplicated().sum()
    if duplicate == 0:
        print('There are no duplicate rows')
    else:
        print(f'There are {duplicate} duplicate rows')

    check_list = {}
    highlight = {}
    cat = df.select_dtypes(include=['O'])
    cat = cat.fillna('NA')

    #  create a list of dataframes of each categorcial column with its unique values
    cat_uniques = [pd.DataFrame(cat[i].unique(), columns=[i]) for i in cat.columns]

    for no, col in enumerate(cat):
        # a dict within a dict
        check_list.update({f'Column {col}':{}})
        highlight.update({f'Column {col}':{}})
        for i in cat_uniques[no].values:
            exclude_self = np.delete(cat_uniques[no].values, np.where(cat_uniques[no].values==i))
            # cat[col].unique()
            match = fuzzywuzzy.process.extract(i[0],exclude_self, limit=limit, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
            check_list[f'Column {col}'].update({i[0]:match})
            for i in match:
                if i[1] >= score:
                    highlight[f'Column {col}'].update({i[0]:match})

        #print(f'{col}: {check_list[col]} \n')
    print("All done!")
    return check_list, highlight


def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

    # let us know the function's done
    print("All done!")


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
