import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix


# Загрузка обработанного датасета
df = pd.read_csv("processed_amazon_sales_dataset.csv")

print("="*70)
print("ЛАБОРАТОРНАЯ РАБОТА №2 - Регрессия и Классификация")
print("="*70)
print(f"\nРазмер датасета: {df.shape}")

# ==========================================
# ЗАДАНИЕ 1: Разделение датасета на выборки
# ==========================================

# Для регрессии предсказываем 'total_revenue'
# Для классификации предсказываем 'rating' (высокий/низкий рейтинг)

X = df.drop(['total_revenue', 'rating'], axis=1)
y_regression = df['total_revenue']

# Бинарная классификация по rating (высокий рейтинг >= медианы)
median_rating = df['rating'].median()
y_classification = (df['rating'] >= median_rating).astype(int)

print(f"\nМедиана rating: {median_rating:.4f}")
print(f"Распределение классов: {y_classification.value_counts().to_dict()}")

# Разделение на выборки
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

print("\n" + "="*70)
print("ЗАДАНИЕ 1: Разделение датасета")
print("="*70)
print(f"Обучающая выборка: {X_train_reg.shape[0]} примеров")
print(f"Тестовая выборка: {X_test_reg.shape[0]} примеров")

# ==========================================
# ЗАДАНИЯ 2-3: Регрессия
# ==========================================

print("\n" + "="*70)
print("ЗАДАНИЕ 2-3: Линейная регрессия для total_revenue")
print("="*70)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_train_pred = reg_model.predict(X_train_reg)
y_test_pred = reg_model.predict(X_test_reg)

train_mse = mean_squared_error(y_train_reg, y_train_pred)
test_mse = mean_squared_error(y_test_reg, y_test_pred)

print(f"\nОбучающая выборка - MSE: {train_mse:.4f}")
print(f"Тестовая выборка - MSE: {test_mse:.4f}")

if test_mse < 0.02:
    print("✓ Хорошая модель!")
else:
    print("⚠️ Требуется улучшение")

# ==========================================
# ЗАДАНИЯ 4-5: Классификация (улучшенная версия)
# ==========================================

print("\n" + "="*70)
print("ЗАДАНИЕ 4-5: Логистическая регрессия для rating")
print("="*70)

# Примечание: данные уже нормализованы (MinMaxScaler из лабы 1),
# поэтому дополнительное масштабирование не требуется

# Метод 1: С балансировкой классов
print("\n--- Метод 1: class_weight='balanced' ---")
clf_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_balanced.fit(X_train_clf, y_train_clf)
y_pred_balanced = clf_balanced.predict(X_test_clf)

acc_balanced = accuracy_score(y_test_clf, y_pred_balanced)
print(f"Accuracy: {acc_balanced:.4f}")
print(classification_report(y_test_clf, y_pred_balanced,
                           target_names=['Низкий рейтинг', 'Высокий рейтинг'],
                           zero_division=0))

# Используем один метод с балансировкой классов
best_acc = acc_balanced
best_model = clf_balanced
best_y_pred = y_pred_balanced
best_y_test = y_test_clf
method_name = "class_weight='balanced'"

print("\n" + "="*70)
print("ЛУЧШАЯ МОДЕЛЬ КЛАССИФИКАЦИИ")
print("="*70)
print(f"Метод: {method_name}")
print(f"Accuracy: {best_acc:.4f}")

conf_matrix = confusion_matrix(best_y_test, best_y_pred)
print(f"\nМатрица ошибок:\n{conf_matrix}")



# ==========================================
# ИТОГИ
# ==========================================

print("\n" + "="*70)
print("ИТОГИ ЛАБОРАТОРНОЙ РАБОТЫ №2")
print("="*70)
print(f"\n1. Регрессия (total_revenue): MSE = {test_mse:.4f}")
print(f"2. Классификация (rating): Accuracy = {best_acc:.4f}")
print(f"\n✓ Все задания выполнены!")
print("="*70)
