import os
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(data_name, data_path, sample_size, seed=42, eda=False):
    '''
    Load data
    Args:
        data_name: dataset name (choices: 'adult', 'bank')
        data_path: path storing dataset
        sample_size: using sample size. if sample_size < 0, then all data is used. otherwise, some samples are sampled ramdomly
        seed: random seed
    Returns:
        data_df: pd.DataFrame
        cat_cols: list of categorical feature names
    '''
    
    if data_name == 'diabetes':
        data_df = pd.read_csv(os.path.join(data_path, data_name, 'diabetic_data.csv'))
        data_df = data_df.drop(['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code'], axis=1) # too many missing value
        data_df = data_df.drop(['diag_3', 'diag_2', 'diag_1'], axis=1) # contains float and str and too many uniq values
        data_df = data_df.drop(['encounter_id', 'patient_nbr'], axis=1) # drop id
        data_df = data_df.drop(['citoglipton', 'examide'], axis=1) # these cols have only 1 value.

        data_df = data_df.drop(['metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 
                                'troglitazone', 'acetohexamide', 
                                ], axis=1) # super imbalance cols
        
        # temporary not included cols: ['glipizide-metformin', 'glyburide-metformin', 'tolazamide', 'miglitol', 'acarbose', 'tolbutamide', 'chlorpropamide' ]
        data_df = data_df.loc[(data_df['glyburide-metformin']!='Up') & (data_df['glyburide-metformin']!='Down')].reset_index(drop=True)
        data_df = data_df.loc[data_df['tolazamide'] != 'Up'].reset_index(drop=True)
        data_df = data_df.loc[(data_df['miglitol']!='Down') & (data_df['miglitol']!='Up')].reset_index(drop=True)
        data_df = data_df.loc[(data_df['acarbose']!='Down') & (data_df['acarbose']!='Up')].reset_index(drop=True)
        data_df = data_df.loc[(data_df['chlorpropamide']!='Down') & (data_df['chlorpropamide']!='Up')].reset_index(drop=True)

        data_df = data_df.loc[data_df['gender'] != 'Unknown/Invalid'].reset_index(drop=True) # There are 3 samples of 'Unknown/Invalid' -> drop

        cat_cols = ['race', 'age', 'number_diagnoses', 
                    'insulin', 'rosiglitazone', 'pioglitazone', 'glyburide', 'glipizide', 'glimepiride','nateglinide', 'repaglinide',
                    'metformin', 
                    ]
        le_cols = ['gender', 'change', 'diabetesMed', 'readmitted',
                   'glipizide-metformin', 'glyburide-metformin', 'tolazamide', 'miglitol', 'acarbose', 'tolbutamide', 'chlorpropamide']

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
        
        # decide target from 'change', 'diabetesMed', 'readmitted'
        data_df.rename(columns={'readmitted':'target'}, inplace=True)

    elif data_name == "adult":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "adult.data"), header=None)
        data_df.rename(columns={0:"age", 1:"workclass", 2:"fnlwgt", 3:"education", 4:"education-num",
                                5:"marital-status", 6:"occupation", 7:"relationship", 8:"race", 9:"sex",
                                10:"capital-gain", 11:"capital-loss", 12:"hours-per-week", 13:"native-country",
                                14:"target"},
                        inplace=True)
        cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]

        #data_df = data_df[cat_cols + ['target', 'sex']]

        # delete rows that have missing values
        data_df = data_df.dropna(how='any')
        # label encoding
        le = LabelEncoder()
        le_cols = ['sex', 'target']
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == "bank":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "bank-full.csv"), sep=';')
        data_df = data_df.rename(columns={"y":"target"})
        data_df = data_df.drop_duplicates().reset_index(drop=True) 

        def month2num(x):
            return str(x).replace('jan','1').replace('feb', '2').replace('mar', '3').replace('apr', '4').replace('may','5').replace('jun', '6').replace('jul', '7').replace('aug', '8').replace('sep', '9').replace('oct','10').replace('nov', '11').replace('dec', '12')

        data_df['month'] = data_df['month'].map(month2num).astype(int)

        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
        le_cols = ['default', 'housing', 'loan', 'target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])
        
        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == "default": 
        data_df = pd.read_excel(os.path.join(data_path, data_name, "default.xls"))
        data_df.columns = data_df.iloc[0]
        data_df = data_df.drop(data_df.index[0])
        data_df.reset_index(drop=True, inplace=True)
        
        data_df = data_df.drop(['ID'], axis=1)
        data_df = data_df.rename(columns={'default payment next month': 'target'})

        cat_cols = ['EDUCATION', 'MARRIAGE']
        le_cols = ['target', 'SEX'] 
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'kick':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'kick.arff'))[0])
        data_df = data_df.rename(columns={'IsBadBuy':'target'})
        data_df = data_df.drop(['WheelTypeID'], axis=1)
        data_df = data_df.dropna(how='any', axis=0) 

        for col in ['BYRNO', 'VNZIP1']:
            data_df[col] = data_df[col].map(lambda x: int(x.decode())) # b'string' -> int

        cat_cols = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'WheelType', 'Nationality',
                    'Size', 'TopThreeAmericanName', 'AUCGUART', 'VNST']

        le_cols = ['target', 'IsOnlineSale', 
                    'PRIMEUNIT', 'Transmission'] # by deleting samples

        # del few patterns
        for cat_col in le_cols + cat_cols:
            few_sample_list = []
            uniq_dict = data_df[cat_col].value_counts().to_dict()
            for uniq_val, num in uniq_dict.items():
                if num < 50:
                    few_sample_list.append(uniq_val)
            data_df = data_df.loc[~data_df[cat_col].isin(few_sample_list)]
            data_df.reset_index(drop=True, inplace=True)

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])
        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == 'census':
        data_df = pd.read_csv(os.path.join(data_path, data_name, 'census-income.data'), header=None,
                            names=['AAGE', 'ACLSWKR', 'ADTINK', 'ADTOCC', 'AHGA', 'AHSCOL', 'AMARITL',
                                    'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT',
                                    'CAPGAIN', 'GAPLOSS', 'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX',
                                    'HHDREL', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP',
                                    'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYYN', 
                                    'WKSWORK', 'income'])
        data_df = data_df.rename(columns={'income':'target'})

        cat_cols = ['ADTINK', 'AHGA', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 'AUNMEM', 'AUNTYPE', 'DIVVAL', 'FILESTAT', 
                    'GRINST', 'HHDFMX', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'NOEMP', 'PENATVTY', 'PEMNTVTY']
        le_cols = ['target', 'AREORGN', 'VETQVA', 'WKSWORK']

        drop_cols = ['GRINREG', 'PARENT', 'PEFNTVTY', 'SEOTR'] # 'SEOTR'はdrop few sample のしきい値を400とした場合

        data_df = data_df.drop(drop_cols, axis=1)

        # del few patterns
        for cat_col in cat_cols + le_cols:
            few_sample_list = []
            uniq_dict = data_df[cat_col].value_counts().to_dict()
            for uniq_val, num in uniq_dict.items():
                if num < 500:
                    few_sample_list.append(uniq_val)
            data_df = data_df.loc[~data_df[cat_col].isin(few_sample_list)]
            data_df.reset_index(drop=True, inplace=True)

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols + cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == 'agrawal':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'agrawal.arff'))[0])
        data_df.rename(columns={'class': 'target'}, inplace=True)

        cat_cols = ['elevel', 'car', 'zipcode']
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])


    else:
        raise NotImplementedError
    
    if sample_size > 0:
        data_df = data_df.sample(n=sample_size, random_state=seed)
        data_df.reset_index(drop=True, inplace=True)

    if eda:
        return data_df, cat_cols, le_cols, num_cols

    return data_df, cat_cols

# for experiments' file name
def weak_cols_code(dataset_name, weak_cols):
    if dataset_name == 'diabetes':
        cat_cols = ['race', 'age', 'number_diagnoses', 
                    'insulin', 'rosiglitazone', 'pioglitazone', 'glyburide', 'glipizide', 'glimepiride','nateglinide', 'repaglinide',
                    'metformin', 
                    ]
    elif dataset_name == 'bank':
        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
    elif dataset_name == 'adult':
        cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]
    elif dataset_name == 'default':
        cat_cols = ['EDUCATION', 'MARRIAGE']
    elif dataset_name == 'kick':
        cat_cols = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'WheelType', 'Nationality',
                    'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'VNST']
    elif dataset_name == 'census':
        cat_cols = ['ADTINK', 'AHGA', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 'AUNMEM', 'AUNTYPE', 'DIVVAL', 'FILESTAT', 
                    'GRINST', 'HHDFMX', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'NOEMP', 'PENATVTY', 'PEMNTVTY']
    elif dataset_name == 'agrawal':
        cat_cols = ['elevel', 'car', 'zipcode']
    else:
        raise NotImplementedError
    
    code = ""
    for cat in cat_cols:
        if cat in weak_cols:
            code += "1"
        else:
            code += "0"
            
    code = int(code, 2)
    
    return code