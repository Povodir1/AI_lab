import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Загрузка данных
df = pd.read_csv("amazon_sales_dataset.csv")

# 1. Количество пропущенных значений для каждого столбца
print("Пропущенные значения:\n", df.isnull().sum())

# 2. Заполнение пропусков
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['object', 'category']:  # Категориальные - мода
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # Числовые - медиана
            df[col].fillna(df[col].median(), inplace=True)

# 3. Нормализация числовых данных
scaler = MinMaxScaler()  # или StandardScaler() для стандартизации
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. One-Hot Encoding для категориальных данных
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nОбработанный датасет:\n", df.head())