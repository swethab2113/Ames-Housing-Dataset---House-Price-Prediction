# Custom transformers for model pipeline using BaseEstimator and TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
class ProductionImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Placeholders for statistics learned ONLY during training
        self.train_medians_ = None
        self.train_modes_ = None
        
    def fit(self, X, y=None):
        # Learn and store the statistics securely from the training slice
        self.train_medians_ = X.median(numeric_only=True)
        self.train_modes_ = X.select_dtypes(include=['object', 'category']).mode().iloc[0]
        return self

    def transform(self, X):
    # Replacing null values
        df_out = X.copy()
        cat_replacements = {
            'Alley': 'No alley access', 
            'MasVnrType': 'None', 
            'BsmtQual': 'No basement',
            'BsmtCond': 'No basement', 
            'BsmtExposure': 'No basement', 
            'BsmtFinType1': 'No basement',
            'BsmtFinType2': 'No basement', 
            'GarageType': 'No garage', 
            'GarageFinish': 'No garage',
            'GarageQual': 'No garage', 
            'GarageCond': 'No garage', 
            'Fence': 'No fence', 
            'MiscFeature': 'No extras'
        }
        df_out.fillna(value=cat_replacements, inplace=True)
        
        if 'GarageYrBlt' in df_out.columns:
            df_out['GarageYrBlt'] = df_out['GarageYrBlt'].fillna(0)
    
        if 'FireplaceQu' in df_out.columns and 'Fireplaces' in df_out.columns:
            df_out.loc[(df_out['FireplaceQu'].isnull()) & (df_out['Fireplaces'] == 0), 'FireplaceQu'] = 'No fireplace'
            df_out['FireplaceQu'] = df_out['FireplaceQu'].fillna('No fireplace')
            
        if 'PoolQC' in df_out.columns and 'PoolArea' in df_out.columns:
            df_out.loc[(df_out['PoolQC'].isnull()) & (df_out['PoolArea'] == 0), 'PoolQC'] = 'No pool'
            df_out['PoolQC'] = df_out['PoolQC'].fillna('No pool')
    
        nan_columns = df_out.columns[df_out.isnull().any()].tolist()
        
        for col in nan_columns:
            if df_out[col].dtype == 'object' or df_out[col].dtype.name == 'category':
                # Use the locked training mode for this column
                fallback_mode = self.train_modes_[col] if col in self.train_modes_ else 'missing'
                df_out[col] = df_out[col].fillna(fallback_mode)
            else:
                # Use the locked training median for this column
                fallback_median = self.train_medians_[col] if col in self.train_medians_ else 0
                df_out[col] = df_out[col].fillna(fallback_median)
                
        return df_out


class ProductionFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No training parameters needed for basic math transformations
        
    def transform(self, X):
        df_out = X.copy()
        
        # House age
        df_out['HouseAge'] = df_out['YrSold'] - df_out['YearBuilt']
        
        # Other rooms above grade
        df_out['OthRmsAbvGrd'] = df_out['TotRmsAbvGrd'] - (df_out['BedroomAbvGr'] + df_out['KitchenAbvGr'])
    
        # House remodeled or not
        df_out['HouseRemod'] = df_out.apply(lambda x: 'Y' if x['YearRemodAdd'] > x['YearBuilt'] else 'N', axis=1)
    
        # Total Porch Sqft
        df_out['TotalPorchSF'] = df_out['WoodDeckSF'] + df_out['OpenPorchSF'] + df_out['EnclosedPorch'] + df_out['3SsnPorch'] + df_out['ScreenPorch']
    
        # Low Quality Fin
        df_out['LowQualFin'] = df_out.apply(lambda x: 'Y' if x['LowQualFinSF'] > 0 else 'N', axis=1)
        
        # Pool existence
        df_out['Pool'] = df_out.apply(lambda x: 'Y' if x['PoolArea'] > 0 else 'N', axis=1)
    
        # Miscellaneous expense existence
        df_out['MiscVal?'] = df_out.apply(lambda x: 'Y' if x['MiscVal'] > 0 else 'N', axis=1)
        return df_out