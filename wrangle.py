import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import acquire
import os
import env
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import env
import acquire
import wrangle

# imports for modeling:
# import Logistic regression
from sklearn.linear_model import LogisticRegression
# import K Nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
# import Decision Trees:
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# import Random Forest:
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

# interpreting our models:
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


directory = os.getcwd()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#------------------------------------------------------------------------------
def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url to access mySQL db
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
#------------------------------------------------------------------------------
def acquire(file = "https://gist.githubusercontent.com/ryanorsinger/\
14c8f919920e111f53c6d2c3a3af7e70/raw/07f6e8004fa171638d6d599cfbf0513f6f60b9e8/student_grades.csv"):
    '''
    acquire will return a dataframe associated with a csv put into the file kwarg.
    it is defaulted to a url pointing to student data.
    '''
    return pd.read_csv(file, index_col=0)
#------------------------------------------------------------------------------
def prepare(df):
    '''
    prepare will take in zillow data, remove any whitespace values
    drop out null values
    and return the entire dataframe.
    '''
    #drop null values
    df = df.dropna()
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df[df.home_value < 1000000]
    county_maps = {6037: 'LA',
    6059: 'Orange',
    6111: 'Ventura'
}
    df['county'] = df.county.map(county_maps)

    return df
#------------------------------------------------------------------------------
def wrangle_zillow():
    SQL_query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017'''
    df = get_zillow_data(SQL_query, directory)
    df = df.dropna()
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.yearbuilt = df.yearbuilt.astype(int)
    df = df.rename(columns={"bedroomcnt": "Bedroom_Count"})
    df = df.rename(columns={"bathroomcnt": "Bathroom_Count"})
    df = df.rename(columns={"calculatedfinishedsquarefeet": "Finished_sqft"})
    df = df.rename(columns={"taxvaluedollarcnt": "Tax_value_dollars"})
    df = df.rename(columns={"yearbuilt": "Year_built"})
    df = df.rename(columns={"taxamount": "Tax_amount"})
    return df
#------------------------------------------------------------------------------
def split_zillow_data(df):
    '''
    This function performs split on zillow data.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
                                        
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test
#------------------------------------------------------------------------------
SQL_query = '''
SELECT bedroomcnt AS bedrooms, 
        bathroomcnt AS bathrooms,
        calculatedfinishedsquarefeet AS total_sqft,
        taxvaluedollarcnt AS home_value,
        fips as county
FROM properties_2017 as pro
JOIN predictions_2017 as pre
	ON pre.id = pro.id
JOIN propertylandusetype as pl
	ON pro.propertylandusetypeid = pl.propertylandusetypeid
WHERE pro.propertylandusetypeid = '261'
'''
#------------------------------------------------------------------------------
def get_zillow_data(SQL_query, directory, filename = 'zillow.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs zillow df
    """
    if os.path.exists(directory+filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_zillow_data(SQL_query)

        df.to_csv(filename)
        return df
#------------------------------------------------------------------------------
def new_zillow_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the zillow db
    """
    url = get_connection_url('zillow')
    
    return pd.read_sql(SQL_query, url)
#------------------------------------------------------------------------------
def scale_data(train, validate, test):

    import sklearn
    from sklearn.preprocessing import MinMaxScaler
    
    #Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    
    
    #Fit the object
    scaler.fit(train[['bedrooms', 'bathrooms', 'total_sqft']])
    
    #Use the object
    scaled_columns = scaler.transform(train[['bedrooms', 'bathrooms', 'total_sqft']])
    
    #Add the newly scaled columns (with the 'Scaled' names) to train
    train[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled']] = scaled_columns
    
    #Make new scaled columns with the newly scaled columns (with the 'Scaled' names) to validate and test

    validate[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled']] = scaler.transform(validate[['bedrooms', 'bathrooms', 'total_sqft']])
    test[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled']] = scaler.transform(test[['bedrooms', 'bathrooms', 'total_sqft']])
    
    train = train.drop(columns=['bedrooms', 'bathrooms', 'total_sqft'])
    validate = validate.drop(columns=['bedrooms', 'bathrooms', 'total_sqft'])
    test = test.drop(columns=['bedrooms', 'bathrooms', 'total_sqft'])
    
    return train, validate, test
    
    
#----------------------------------------------------------------------------------------------------------------------------------   
    
def Qscale_data(train, validate, test):  
    
    import sklearn
    from sklearn.preprocessing import QuantileTransformer
    
    #Create the object
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    
    #Fit the object
    scaler.fit(train[['Bedroom_Count', 'Bathroom_Count', 'Finished_sqft', 'Tax_value_dollars', 'Tax_amount', 'Year_built']])
    
    #Use the object
    scaled_columns = scaler.transform(train[['Bedroom_Count', 'Bathroom_Count', 'Finished_sqft', 'Tax_value_dollars', 'Tax_amount', 'Year_built']])
    
    #Add the newly scaled columns (with the 'Scaled' names) to train
    train[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled', 'Tax_value_dollars_Scaled', 'Tax_amount_Scaled', 'Year_built_Scaled']] = scaled_columns

    #Make new scaled columns with the newly scaled columns (with the 'Scaled' names) to validate and test

    validate[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled', 'Tax_value_dollars_Scaled', 'Tax_amount_Scaled', 'Year_built_Scaled']] = scaler.transform(validate[['Bedroom_Count', 'Bathroom_Count', 'Finished_sqft', 'Tax_value_dollars', 'Tax_amount', 'Year_built']])
    test[['Bedroom_Count_Scaled', 'Bathroom_Count_Scaled', 'Finished_sqft_Scaled', 'Tax_value_dollars_Scaled', 'Tax_amount_Scaled','Year_built_Scaled']] = scaler.transform(test[['Bedroom_Count', 'Bathroom_Count', 'Finished_sqft', 'Tax_value_dollars', 'Tax_amount', 'Year_built']])
    
    return train, validate, test
#------------------------------------------------------------------------------
def plot_variable_pairs(df):
    train, validate, test = split_zillow_data(df)
    X_train = train.drop(columns='Tax_value_dollars')
    y_train = train[['Tax_value_dollars']]
    
    
    homes_corr = train.drop(columns=['fips']).corr()
    plt.figure(figsize=(16, 3))
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(homes_corr, kind="reg", corner=True, plot_kws={'line_kws':{'color':'red'}})
    plt.show()
    
    #Let's look at the features through a heatmap
    homes_corr = train.corr()
    
    # Pass my correlation matrix to Seaborn's heatmap.
    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
              'linecolor':'k','rasterized':False, 'edgecolor':'w', 
              'capstyle':'projecting',}
    
    plt.figure(figsize=(8,6))
    sns.heatmap(homes_corr, cmap='Purples', annot=True, mask= np.triu(homes_corr), **kwargs)
    plt.ylim(0, 7)
    
    plt.show()
    #------------------------------------------------------------------------------
def plot_categorical_and_continuous_vars(df):
    train, validate, test = split_zillow_data(df)
    X_train = train.drop(columns='Tax_value_dollars')
    y_train = train[['Tax_value_dollars']]
    
    for col in X_train:
        print(f'Distribution of {col}')
        sns.boxplot(data=X_train, x=col)
        plt.show()
        plt.hist(X_train[col])
        plt.show()
        sns.relplot(x=X_train[col], y="Tax_value_dollars", data=train, markers='.')
        plt.show()
        
    
    #------------------------------------------------------------------------------
def evaluate_reg(y, yhat):
    '''
    based on two series, y_act, y_pred, (y, yhat), we
    evaluate and return the root mean squared error
    as well as the explained variance for the data.
    
    returns: rmse (float), rmse (float)
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2
    
    #------------------------------------------------------------------------------
def boxplot(df):
    df1 = df.drop(columns={'home_value', 'county'})
    for col in df1:
        plt.figure(figsize=(12, 10))
        plt.subplot(222)
        sns.boxplot(data=df, x=col)
        plt.title(f'Boxplot of {col}')
    
    #------------------------------------------------------------------------------
def heatmap(train):
    train_corr = train.corr()

    # Pass my correlation matrix to Seaborn's heatmap.
    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
              'linecolor':'k','rasterized':False, 'edgecolor':'w', 
              'capstyle':'projecting',}
    
    plt.figure(figsize=(8,6))
    sns.heatmap(train_corr, cmap='Purples', annot=True, mask= np.triu(train_corr), **kwargs)
    plt.ylim(0, 4)
    
    plt.show()
    
#------------------------------------------------------------------------------
def scatterplots(train):
    # Scatterplot with a regression line
    sns.lmplot(x="bedrooms", y="home_value", data=train, line_kws={'color': 'red'})
    plt.show()
    sns.lmplot(x="bathrooms", y="home_value", data=train, line_kws={'color': 'red'})
    plt.show()
    sns.lmplot(x="total_sqft", y="home_value", data=train, line_kws={'color': 'red'})
    plt.show()
#------------------------------------------------------------------------------
def boxplots(train):
    sns.boxplot(data = train, y = train.total_sqft, x = train.home_value)
    plt.show()
    sns.boxplot(data = train, y = train.home_value, x = train.bathrooms)
    plt.show()
    sns.boxplot(data = train, y = train.home_value, x = train.bedrooms)
    plt.show()
    sns.boxplot(data = train, y = train.bedrooms, x = train.bathrooms)
    plt.show()
#------------------------------------------------------------------------------
def barplots(train):
    train1 = train.drop(columns='home_value')
    for col in train1:
        plt.figure(figsize=(20,12))
        plt.title(f"Comparision between {col} and home value")
        sns.barplot(x=col, y="home_value", data=train)
        home_value_average = train.home_value.mean()
        plt.axhline(home_value_average, label="Home Value Average")
        plt.legend()
        plt.show()
#------------------------------------------------------------------------------
def Correlation_Total_Sqft_Home_Value(train):
    plt.figure(figsize=(20,12))
    sns.lmplot(x="total_sqft", y="home_value", data=train, line_kws={'color': 'red'})
    plt.title('Correlation in Total_Sqft & Home Value') #title
    plt.xlabel('Total Square Feet') #x label
    plt.ylabel('Home Value') #y label
    plt.show()
#------------------------------------------------------------------------------
def Predicting_Home_Value_via_Bedrooms(train):
        plt.figure(figsize=(20,8))
        plt.title(f"Comparision between bedrooms and home value")
        sns.barplot(x='bedrooms', y="home_value", data=train)
        home_value_average = train.home_value.mean()
        plt.axhline(home_value_average, label="Home Value Average")
        plt.legend()
        plt.show()
#------------------------------------------------------------------------------

def Predicting_Home_Value_via_Bathrooms(train):
    plt.figure(figsize=(20,8))
    plt.title(f"Comparision between bathrooms and home value")
    sns.boxplot(data = train, y = train.home_value, x = train.bathrooms)
    plt.show()
    
#------------------------------------------------------------------------------
def Predicting_Home_Value_per_County(train):
    plt.figure(figsize=(16,8))
    plt.title(f"Comparision between county and home value")
    sns.barplot(x='county', y="home_value", data=train)
    home_value_average = train.home_value.mean()
    plt.axhline(home_value_average, label="Home Value Average")
    plt.legend()
    plt.show()
#------------------------------------------------------------------------------
def XandY_train_validate_test(train, validate, test):
    #Let's split up the train data between X and y
    X_train_scaled = train.drop(columns={'home_value'})
    y_train_scaled = train['home_value']
    X_validate_scaled = validate.drop(columns={'home_value'})
    y_validate_scaled = validate['home_value']
    X_test_scaled = test.drop(columns={'home_value'})
    y_test_scaled = test['home_value']
    return X_train_scaled, y_train_scaled, X_validate_scaled, y_validate_scaled, X_test_scaled, y_test_scaled
#------------------------------------------------------------------------------
def encode_data(train, validate, test):
    county_maps = {'LA': 1,
    'Orange': 2,
    'Ventura': 3
}

    train['county'] = train.county.map(county_maps)
    validate['county'] = validate.county.map(county_maps)
    test['county'] = test.county.map(county_maps)

    return train, validate, test
#------------------------------------------------------------------------------
def eval_dist(r, p, α=0.05):
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")
#------------------------------------------------------------------------------
def eval_Spearmanresult(r,p,α=0.05):
    """
    
    """
    if p < α:
        return print(f"""We reject H₀, there appears to be a monotonic relationship.
Spearman's rs: {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there does not appear to be a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")
#------------------------------------------------------------------------------
def XandY(train, validate, test):
    #Let's split up the train data between X and y
    X_train = train.drop(columns={'home_value'})
    y_train = train['home_value']
    X_validate = validate.drop(columns={'home_value'})
    y_validate = validate['home_value']
    X_test = test.drop(columns={'home_value'})
    y_test = test['home_value']
    return X_train, y_train, X_validate, y_validate, X_test, y_test
#------------------------------------------------------------------------------

def select_Kbest(X_train_scaled, y_train):
    from sklearn.feature_selection import SelectKBest, f_regression

    # parameters: f_regression stats test, give me the top feature
    f_selector = SelectKBest(f_regression, k=2)
    
    # find the top X correlated with y
    f_selector.fit(X_train_scaled, y_train)
    
    # save top features
    f_features = f_selector.get_feature_names_out()
    return f_features
#------------------------------------------------------------------------------
def get_RFE(X_train_scaled,y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    
    # initialize the ML algorithm
    lm = LinearRegression()
    
    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=2)
    
    # fit the data using RFE
    rfe.fit(X_train_scaled,y_train)  
    
    # get list of the column names. 
    rfe_feature = rfe.get_feature_names_out()
    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Variable': var_names, 'Rank': var_ranks})
    # sort the df by rank
    k = rfe_ranks_df.sort_values('Rank').head()
    
    return rfe_feature, k
#------------------------------------------------------------------------------
def baseline(X_train_scaled, y_train_scaled):
    # X -> set of all ind features
    #y -> dependent variable, target
    X = X_train_scaled
    y = y_train_scaled
    
    from sklearn.linear_model import LinearRegression

    #make
    lm = LinearRegression()
    #fit
    lm.fit(X,y)
    #use
    yhat = lm.predict(X)
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    baseline_med = y.median()
    baseline_mean = y.mean()
    
    # compute the error on these two baselines:
    mean_baseline_rmse = mean_squared_error(y_pred.baseline_mean, y) ** (1/2)
    med_baseline_rmse = mean_squared_error(y_pred.baseline_med, y) ** (1/2)
    
    #Establish the true baseline
    baseline = mean_baseline_rmse
    
    # Squared Errors, 
    # residuals squared
    squared_errors = (y_pred['y_act'] - y_pred['yhat']) ** 2
    # Sum of Squared Error, 
    sse = squared_errors.sum()
    # Mean Squared error, 
    mse = sse / train.shape[0] #alternatively, len(train), y_pred.shape[0]
    # Root mean Squared Error
    # square root the mean squared error!
    rmse = mse**0.5
    
    # Squared Errors, 
    # residuals squared
    squared_errors_bl = (y_pred['y_act'] - y_pred['baseline_mean']) ** 2
    # Sum of Squared Error, 
    sse_bl = squared_errors_bl.sum()

    
    return baseline


    #------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
