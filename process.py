from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
class DataPreprocessor:
    def __init__(self, numeric_features=None, categorical_features=None,
                 binary_features=None, ordinal_cat_features=None, ordinal_num_features=None,
                 ordinal_categories=None,use_smote=True):
        """
        Inicializa o pré-processador com listas de colunas e categorias ordinais.

        Parâmetros:
        - numeric_features: Lista de colunas numéricas contínuas.
        - categorical_features: Lista de colunas categóricas nominais (para OneHotEncoder).
        - binary_features: Lista de colunas binárias (ex.: Sim/Não).
        - ordinal_cat_features: Lista de colunas categóricas ordinais.
        - ordinal_num_features: Lista de colunas numéricas ordinais (ex.: notas 1 a 5).
        - ordinal_categories: Lista de listas com categorias ordenadas para ordinal_cat_features.
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.binary_features = binary_features or []
        self.ordinal_cat_features = ordinal_cat_features or []
        self.ordinal_num_features = ordinal_num_features or []
        self.ordinal_categories = ordinal_categories or []
        self.pipeline = None
        self.use_smote = use_smote


    def build_pipeline(self):
        """
        Constrói o ColumnTransformer com pipelines específicos para cada tipo de coluna.
        """
        transformers = []

        # Pipeline para colunas numéricas contínuas
        if self.numeric_features:
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, self.numeric_features))

        # Pipeline para colunas categóricas nominais
        if self.categorical_features:
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_features))

        # Pipeline para colunas binárias
        if self.binary_features:
            binary_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('bin', binary_pipeline, self.binary_features))

        # Pipeline para colunas categóricas ordinais
        if self.ordinal_cat_features:
            if len(self.ordinal_categories) != len(self.ordinal_cat_features):
                raise ValueError("Número de listas de categorias ordinais deve corresponder ao número de colunas ordinais.")
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=self.ordinal_categories,
                                         handle_unknown='use_encoded_value',
                                         unknown_value=-1))
            ])
            transformers.append(('ord_cat', ordinal_pipeline, self.ordinal_cat_features))

        # Pipeline para colunas numéricas ordinais (mantidas como estão)
        if self.ordinal_num_features:
            ordinal_num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', 'passthrough')  # Mantém valores como estão
            ])
            transformers.append(('ord_num', ordinal_num_pipeline, self.ordinal_num_features))

        # Criar o ColumnTransformer
        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Mantém colunas não especificadas sem transformação
        )

    def apply_smote(self, X, y):
        """
        Aplica SMOTE para balanceamento de classes, se habilitado.

        Parâmetros:
        - X: Dados de entrada transformados.
        - y: Rótulos/classes correspondentes.

        Retorna:
        - X_resampled, y_resampled: Dados após o SMOTE.
        """
        if self.use_smote:
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled


    def data_transformer(self, data, fit=True):
        """
        Aplica a transformação nos dados com base nos tipos de colunas definidos.

        Parâmetros:
        - data: DataFrame a ser transformado.
        - fit: Se True, aplica fit_transform; caso contrário, apenas transform.

        Retorna:
        - array numpy com os dados transformados.
        """
        self.build_pipeline()
        if fit:
            return self.pipeline.fit_transform(data)
        else:
            return self.pipeline.transform(data)

