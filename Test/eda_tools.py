import pandas as pd
import numpy as np
import re

from scipy.stats import spearmanr, pearsonr

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

import fuzzywuzzy
from fuzzywuzzy import process


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
    #duplicate = df.duplicated().sum()
    #if duplicate == 0:
    #    print('There are no duplicate rows')
    #else:
    #    print(f'There are {duplicate} duplicate rows')

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





#####################################
# apply SKLEARN
#####################################

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
    fig.show()
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

#####################################
# Time base
#####################################

def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
'''
# Cell
def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a
 '''
def ifnone(a, b):
    return b if a is None else a


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


###################################
# Plots
###################################

def plot_data_shape(col):
    '''
    This function will plot a histgram and a boxplot side by side
    It accepts 2 arguments:
    Pass a dataframe into data
    and the column name you wish to examine into col

    data: the dataframe
    col: name of the column to be plotted
    
    '''
    
    assert col.dtype!='O', 'column passed in must be numeric'
    sns.set_style("darkgrid")
    sns.set_color_codes(palette = 'deep')
    fig , ax = plt.subplots(2,1,figsize=(12,6), dpi =80)
    
    axe = plt.twinx()
    axe.set_ylabel('density')
    
    try:
        name = col.columns.values
    except:
        name = col.name
    else:
        name = 'numpy array'
        
    ax[0].set_title(f'boxplot: {name}')
    sns.boxplot(data= col,showmeans=True,notch=True, ax=ax[0],orient='h', color='darkmagenta', flierprops={"marker":"o",
                       "markerfacecolor":"orange", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
    
    ax[1].set_title(f'Histogram: {name}')
    sns.histplot(data= col, ax=ax[1], kde=True, color= 'darkmagenta')
    

def plot_shape_transform(col, transform='log'):
    '''
    plot out a histogram on before and after application of transform 
    col : accepts a column of values 
    transform : type of transformation to be applied
    '''
    try:
        skew = col.skew()
        kurtosis = col.kurtosis()
    except:
        skew= 'NA'
        kurtosis = 'NA'
    
    if transform == 'log':
        col_t = np.log(col)
    
    elif transform == 'log1p':
        col_t = np.log1p(col)
    
    elif transform == 'sqrt':
        col_t = col**0.5
    
    if skew !='NA':
        skew_t = col_t.skew()
        kurtosis_t = col_t.kurtosis()
        
    fig, axs = plt.subplots(1, 2,figsize=(12, 6), dpi=80)
    fig.suptitle(f'Comparison of Variable after {transform} scale', fontsize=16)
    axs[0].set_title(f'Shape of Original Variable\n Skew: {skew:.3f} |  Kurtosis: {kurtosis:.3f}')
    sns.histplot(col, kde=True,color='darkmagenta',ax=axs[0])
    axs[0].axvline(col.mean(), color='g', linestyle='--', linewidth=3)
    axs[1].set_title(f'Shape of Variable after {transform} scale\n Skew: {skew_t:.3f} |  Kurtosis: {kurtosis_t:.3f}')
    sns.histplot(col_t, kde=True,color='darkmagenta', ax=axs[1])
    axs[1].axvline(x=col_t.mean(), color='g', linestyle='--', linewidth=3)
    
    
def plot_feature_outlier(data, col,iqr=None,low_z=True,high_z=True, low_fence=True, high_fence=True, inner_fence=False, z=False):
    '''
    create a histogram and box plot for comparsion in order to discern outliers for a feature.
    
    data: the dataframe
    col: the column in question to be examined
    iqr: to set the interquartle range which will highlight the outliers
    low_z: draw the lower boundary line for z-score in the histogram
    high_z: draw the upper boundary line for z-score in the histogram
    low_fence: draw the lower boundary for interquartile range in the boxplot
    high_fence: draw the upper boundary for interquartile range in the boxplot
    inner_fence: to highlight the 1.5x interquartile range 
    z: when True the histogram will be ase on z-score instead of col

    '''   
    
    mean = data[col].mean()
    min = data[col].min()
    max = data[col].max()
    std = data[col].std()
    print(f'Mean of {col}                     :{mean}')
    print(f'Min of {col}                      :{min}')
    print(f'Max of {col}                      :{max}')
    print(f'Standard Deviation of {col}       :{std}')
    zscore = (data[[col]] - data[[col]].mean())/data[[col]].std()
    zscore = zscore.rename(columns={col: 'z-score'})

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    print(f'The interquartile range for {col} :{IQR}')

    fig, ax = plt.subplots(2,1, figsize=(10,10))
    fig.tight_layout(pad=6.0)
    fig.suptitle('Outliers detection using z-score and Interquartile Range', fontsize=16)
    
    #sns.histplot(data[col], color='purple', kde=True,ax=ax[0])
    # Histplot of feature
    if z:
        print('\nCurrent view is z=True ; relative view')
        sns.histplot(zscore['z-score'], color='darkmagenta',ax=ax[0])
        ax[0].set_xlim(-3.5,  3.5)
        axe = ax[0].twinx()
        sns.kdeplot(zscore['z-score'],fill=True,ax=axe)
        if high_z:
            ax[0].axvline(x=3, color='#FF5959', linewidth=3, label= r'$\sigma$ >= 3')    
        if low_z:
            ax[0].axvline(x=-3, color='#FF5959', linewidth=3, label= r'$\sigma$ <= 3')
            
    else:
        print('\nCurrent view is z=False ; x-axis set to max 3.5 interquatile range')
        sns.histplot(data[col], color='darkmagenta',ax=ax[0])
        ax[0].set_xlim( Q1  - (3.5* IQR),  Q3  + (3.5* IQR))
        ax[1].set_xlim( Q1  - (3.5* IQR),  Q3  + (3.5* IQR))
        if high_z:
            ax[0].axvline(x=mean+3*std, color='#FF5959', linewidth=3, label= r'$\sigma$ >= 3')
        if low_z:
            ax[0].axvline(x=mean-3*std, color='#FF5959', linewidth=3, label= r'$\sigma$ <= 3')
       
   
  
    ax[0].set_title('Outliers usually lies beyond z-score of 3')
    ax[0].legend(fontsize =12)
    if z:
        ax[0].set_xlabel(r'$\sigma$', fontsize =18)




    # Boxplot of feature 
    sns.boxplot(data=data[[col]] ,x = col,color='darkmagenta',whis=iqr ,ax=ax[1], flierprops={"marker":"o",
                       "markerfacecolor":"#FAEDC6", 
                       "markeredgecolor":"#F3C892",
                      "markersize":"10"})

    if high_fence:
        if inner_fence:
            ax[1].axvline(x= Q3 + (1.5* IQR), color='#FABB51', linewidth=3, label="lnner Fence> 1.5 IQR")
        ax[1].axvline(x= Q3  + (3* IQR), color='#FF5959', linewidth=3,  label="Outer Fence > 3 IQR")
    if low_fence:
        if inner_fence:
            ax[1].axvline(x= Q1 - (1.5* IQR), color='#FABB51', linewidth=3, label="Inner Fence < 1.5 IQR")
        ax[1].axvline(x= Q1  - (3* IQR), color='#FF5959', linewidth=3, label="Outer Fence < 3 IQR")
    

    ax[1].set_title('Major Outliers lie beyond red line, Minor Outliers in the range between yellow and red')
    ax[1].legend(fontsize=12,loc='best')
    plt.show()
    
    
###############################
# Stats
###############################

def get_corrs(data,y,method='pearson', round=8):
    
    '''
        data: Pass in a dataframe
        y: the variable to compare with
        method: pearson , spearman 
        
    '''
    
    dic = {'dependent':[],'feature':[], 'correlation':[],'p-values':[]}
    for col in data.columns.values:
        if method == 'pearson':
            values, pv = pearsonr(data[data[col].notnull()][col], data.loc[data[data[col].notnull()].index.values,y])
        elif method == 'spearman':
            values, pv = spearmanr(data[data[col].notnull()][col],data.loc[data[data[col].notnull()].index.values,y])
        dic['dependent'].append(y)
        dic['feature'].append(col)
        dic['correlation'].append(values)
        dic['p-values'].append(pv)
    return pd.DataFrame.from_dict(dic).round(round)
    