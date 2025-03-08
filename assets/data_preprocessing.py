from assets.imports import *
from ucimlrepo import fetch_ucirepo 

def preprocess_data():
    # fetch dataset 
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    # data (as pandas dataframes) 
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets 

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(X[categorical_cols])

    # Convert encoded data to DataFrame
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    X = pd.concat([X, encoded_categorical_df], axis=1)

    return X, y