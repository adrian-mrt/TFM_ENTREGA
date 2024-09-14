# transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import pandas as pd

class CleanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Reemplazo de GarageYrBlt
        df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
        
        # Cambio de MSZoning
        df['MSZoning'] = df['MSZoning'].replace('C (all)', 'C') 
        
        # Cambio de BldgType
        df['BldgType'] = df['BldgType'].replace({'2fmCon': '2FmCon', 'Twnhs': 'TwnhsI'})
        
        # Cambio de Exterior1st
        df["Exterior1st"] = df["Exterior1st"].replace({"Wd Sdng": "WdSdng"})
        
        # Cambio de Exterior2nd
        df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
        df["Exterior2nd"] = df["Exterior2nd"].replace({"Wd Sdng": "WdSdng"})
        df["Exterior2nd"] = df["Exterior2nd"].replace({"Wd Shng": "WdShing"})
        
        # Renombrar columnas
        df.rename(columns={
            "1stFlrSF": "FirstFlrSF",
            "2ndFlrSF": "SecondFlrSF",
            "3SsnPorch": "ThreeSeasonPorch",
        }, inplace=True)
        
        return df

class EncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_vars, ordered_levels):
        self.nominal_vars = nominal_vars
        self.ordered_levels = ordered_levels
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        # Codificación de variables nominales
        for var in self.nominal_vars:
            df[var] = df[var].astype("category")
            if "None" not in df[var].cat.categories:
                df[var] = df[var].cat.add_categories("None")
        
        # Ordenar categorías
        for var, levels in self.ordered_levels.items():
            df[var] = df[var].astype(CategoricalDtype(levels, ordered=True))
        
        return df
    
class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        for name in df.select_dtypes("number"):
            df[name] = df[name].fillna(0)
        for name in df.select_dtypes("category"):
            df[name] = df[name].fillna("None")
        return df
    
class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_dict = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.le_dict[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            le = self.le_dict[col]
            X_transformed[col] = le.transform(X_transformed[col])
        return X_transformed

    
# Transformer PCA 
class PCAToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # PCA
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X)
        # Nombre de columnas
        column_names = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_pca, columns=column_names)