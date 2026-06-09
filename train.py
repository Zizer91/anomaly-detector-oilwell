"""
Автоматический подбор гиперпараметров для детектора аномалий Isolation Forest.
Перебирает window_size, step, contamination и вычисляет recall.
Тестирует модель на разных типах аномалий.
Экспериментирует с разными наборами признаков.
"""

from pathlib import Path
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score
import joblib
from src.data_preprocessor import DataPreprocessor

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Параметры для перебора
WINDOW_SIZES = [30, 60, 120, 180]      # размер окна в секундах
STEPS = [15, 30, 60]                    # шаг сдвига окна
CONTAMINATIONS = [0.01, 0.03, 0.05, 0.1]  # ожидаемая доля аномалий

# Признаки для обучения (можно добавлять новые)
BASE_FEATURES = ['P-PDG', 'P-TPT', 'T-PDG', 'T-TPT']
PRESSURE_FEATURES = ['P-ANULAR', 'P-JUS-BS', 'P-JUS-CKGL', 'P-JUS-CKP',
                     'P-MON-CKGL', 'P-MON-CKP', 'P-MON-SDV-P']
RATE_FEATURES = ['QBS', 'QGL']
TEMP_FEATURES = ['T-JUS-CKP', 'T-MON-CKP']
STATE_FEATURES = ['ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-M2', 'ESTADO-PXO',
                  'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1', 'ESTADO-W2', 'ESTADO-XO']
CHOKE_FEATURES = ['ABER-CKGL', 'ABER-CKP']

# Используемый набор признаков (можно менять)
FEATURE_COLS = BASE_FEATURES+TEMP_FEATURES

# Пути к файлам (настрой под свою структуру)
NORMAL_FOLDER = Path(r'D:\InterWells\dataset\0')
ANOMALY_FILE = Path(r'D:\InterWells\dataset\5\SIMULATED_00001.parquet')

# ============================================================
# ЗАГРУЗКА ДАННЫХ (один раз, для всех экспериментов)
# ============================================================

print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

# Загружаем все нормальные файлы из папки 0
normal_files = list(NORMAL_FOLDER.glob('*.parquet'))
print(f"Найдено файлов с нормальной работой: {len(normal_files)}")

# Загружаем нормальные данные (только класс 0)
normal_dfs = []
for f in normal_files:
    df = pd.read_parquet(f)
    df_normal = df[df['class'] == 0].copy()
    if len(df_normal) > 0:
        normal_dfs.append(df_normal)
        print(f"  {f.name}: {len(df_normal)} строк")
    else:
        print(f"  {f.name}: нет данных класса 0")

X_normal_raw = pd.concat(normal_dfs, ignore_index=True)
print(f"\nВсего строк нормальной работы: {len(X_normal_raw)}")

# Загружаем аномальный файл для подбора параметров
df_anomaly_raw = pd.read_parquet(ANOMALY_FILE)
print(f"Аномальный файл: {len(df_anomaly_raw)} строк")

# ============================================================
# АВТОМАТИЧЕСКИЙ ПЕРЕБОР ПАРАМЕТРОВ
# ============================================================

print("\n" + "=" * 70)
print("ПЕРЕБОР ПАРАМЕТРОВ")
print("=" * 70)
print(f"{'ws':>4} | {'step':>4} | {'cont':>5} | {'recall':>6} | {'precision':>9} | {'f1':>6}")
print("-" * 70)

best_recall = -1
best_params = None
best_metrics = {}

for ws, step, cont in itertools.product(WINDOW_SIZES, STEPS, CONTAMINATIONS):
    # Создаём препроцессор с текущими параметрами
    prep = DataPreprocessor(window_size=ws, step=step)

    # Нарезаем нормальные данные на окна
    X_train = prep.create_windows(X_normal_raw, FEATURE_COLS)

    # Нарезаем аномальный файл на окна
    X_test = prep.create_windows(df_anomaly_raw, FEATURE_COLS)

    # Если после нарезки не осталось окон — пропускаем
    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Обучаем детектор
    detector = IsolationForest(contamination=cont, random_state=42)
    detector.fit(X_train.reshape(X_train.shape[0], -1))

    # Предсказываем (1 = аномалия, 0 = норма)
    preds = detector.predict(X_test.reshape(X_test.shape[0], -1))
    pred_labels = [1 if p == -1 else 0 for p in preds]

    # Метки реальных аномалий: если в окне есть хотя бы одна точка с class == 5
    true_labels = []
    for i in range(0, len(df_anomaly_raw) - ws + 1, step):
        window_classes = df_anomaly_raw['class'].iloc[i:i+ws]
        true_labels.append(1 if (window_classes == 5).any() else 0)

    # Обрезаем до одинаковой длины
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    # Считаем метрики
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print(f"{ws:4d} | {step:4d} | {cont:5.2f} | {recall:6.3f} | {precision:9.3f} | {f1:6.3f}")

    if recall > best_recall:
        best_recall = recall
        best_params = {'window_size': ws, 'step': step, 'contamination': cont}
        best_metrics = {'recall': recall, 'precision': precision, 'f1': f1}

# ============================================================
# РЕЗУЛЬТАТЫ ПОДБОРА
# ============================================================

print("\n" + "=" * 70)
print("ЛУЧШИЕ ПАРАМЕТРЫ")
print("=" * 70)
print(f"window_size   : {best_params['window_size']}")
print(f"step          : {best_params['step']}")
print(f"contamination : {best_params['contamination']}")
print(f"\nМетрики на лучших параметрах:")
print(f"  Recall    : {best_metrics['recall']:.3f}")
print(f"  Precision : {best_metrics['precision']:.3f}")
print(f"  F1-score  : {best_metrics['f1']:.3f}")

# ============================================================
# ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ С ЛУЧШИМИ ПАРАМЕТРАМИ
# ============================================================

print("\n" + "=" * 70)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 70)

prep_final = DataPreprocessor(
    window_size=best_params['window_size'],
    step=best_params['step']
)

X_train_final = prep_final.create_windows(X_normal_raw, FEATURE_COLS)

detector_final = IsolationForest(
    contamination=best_params['contamination'],
    random_state=42
)
detector_final.fit(X_train_final.reshape(X_train_final.shape[0], -1))

# Сохраняем модель
joblib.dump(detector_final, 'anomaly_detector.pkl')
print("Модель сохранена в 'anomaly_detector.pkl'")

# ============================================================
# ТЕСТИРОВАНИЕ НА РАЗНЫХ ТИПАХ АНОМАЛИЙ
# ============================================================

print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ НА РАЗНЫХ ТИПАХ АНОМАЛИЙ")
print("=" * 70)

# Словарь с путями к файлам для разных типов аномалий
anomaly_files = {
    'Class 1 (BSW)': Path(r'D:\InterWells\dataset\1\SIMULATED_00001.parquet'),
    'Class 5 (Prod Loss)': ANOMALY_FILE,
    'Class 6 (PCK Restriction)': Path(r'D:\InterWells\dataset\6\SIMULATED_00001.parquet'),
    'Class 7 (Scaling)': Path(r'D:\InterWells\dataset\7\DRAWN_00001.parquet'),
    'Class 8 (Hydrate Prod)': Path(r'D:\InterWells\dataset\8\SIMULATED_00001.parquet'),
    'Class 9 (Hydrate Service)': Path(r'D:\InterWells\dataset\9\SIMULATED_00001.parquet'),
}

print(f"{'Тип аномалии':<25} | {'Recall':>6} | {'Precision':>9} | {'F1':>6}")
print("-" * 70)

for name, file_path in anomaly_files.items():
    if not file_path.exists():
        print(f"{name:<25} | {'Файл не найден':>6}")
        continue

    # Загружаем файл
    df_test = pd.read_parquet(file_path)

    # Нарезаем окна
    X_test = prep_final.create_windows(df_test, FEATURE_COLS)

    if len(X_test) == 0:
        print(f"{name:<25} | {'Нет данных':>6}")
        continue

    # Предсказываем
    preds = detector_final.predict(X_test.reshape(X_test.shape[0], -1))
    pred_labels = [1 if p == -1 else 0 for p in preds]

    # Реальные метки (если в окне есть аномальный класс != 0)
    true_labels = []
    for i in range(0, len(df_test) - best_params['window_size'] + 1, best_params['step']):
        window_classes = df_test['class'].iloc[i:i+best_params['window_size']]
        is_anomaly = 1 if (window_classes != 0).any() else 0
        true_labels.append(is_anomaly)

    # Обрезаем до одинаковой длины
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    if sum(true_labels) == 0:
        print(f"{name:<25} | {'Нет аномалий':>6}")
        continue

    recall = recall_score(true_labels, pred_labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print(f"{name:<25} | {recall:6.3f} | {precision:9.3f} | {f1:6.3f}")

# ============================================================
# ЭКСПЕРИМЕНТЫ С РАЗНЫМИ НАБОРАМИ ПРИЗНАКОВ
# ============================================================

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТЫ С РАЗНЫМИ НАБОРАМИ ПРИЗНАКОВ")
print("=" * 70)

feature_sets = {
    'Базовые (P, T)': BASE_FEATURES,
    'Базовые + дебиты': BASE_FEATURES + RATE_FEATURES,
    'Базовые + штуцеры': BASE_FEATURES + CHOKE_FEATURES,
    'Базовые + давления': BASE_FEATURES + PRESSURE_FEATURES,
    'Базовые + температуры': BASE_FEATURES + TEMP_FEATURES,
    'Базовые + состояния': BASE_FEATURES + STATE_FEATURES,
}

# Загружаем аномальный файл ещё раз (для экспериментов)
df_test_raw = pd.read_parquet(ANOMALY_FILE)

for set_name, features in feature_sets.items():
    # Проверяем, что все признаки есть в данных
    available_features = [f for f in features if f in X_normal_raw.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"{set_name}: пропущены признаки {missing} (нет в данных)")
        features = available_features

    if len(features) == 0:
        print(f"{set_name}: нет доступных признаков")
        continue

    # Нарезаем данные с новым набором признаков
    X_train_feat = prep_final.create_windows(X_normal_raw, features)
    X_test_feat = prep_final.create_windows(df_test_raw, features)

    if len(X_train_feat) == 0 or len(X_test_feat) == 0:
        print(f"{set_name}: недостаточно данных для окон")
        continue

    # Обучаем модель
    detector_test = IsolationForest(contamination=best_params['contamination'], random_state=42)
    detector_test.fit(X_train_feat.reshape(X_train_feat.shape[0], -1))

    # Оцениваем
    preds = detector_test.predict(X_test_feat.reshape(X_test_feat.shape[0], -1))
    pred_labels = [1 if p == -1 else 0 for p in preds]

    # True labels для аномального файла (класс 5)
    true_labels = []
    for i in range(0, len(df_test_raw) - best_params['window_size'] + 1, best_params['step']):
        window_classes = df_test_raw['class'].iloc[i:i+best_params['window_size']]
        true_labels.append(1 if (window_classes == 5).any() else 0)

    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    if sum(true_labels) == 0:
        print(f"{set_name}: нет аномалий класса 5 в тестовых данных")
        continue

    recall = recall_score(true_labels, pred_labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print(f"{set_name:<30} | recall={recall:.3f}, precision={precision:.3f}, f1={f1:.3f}")

# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

print("\n" + "=" * 70)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Сохраняем параметры и метрики
results = {
    'best_params': best_params,
    'best_metrics': best_metrics,
    'feature_cols': FEATURE_COLS,
}

with open('anomaly_detection_results.txt', 'w') as f:
    f.write("ЛУЧШИЕ ПАРАМЕТРЫ\n")
    f.write(f"window_size: {best_params['window_size']}\n")
    f.write(f"step: {best_params['step']}\n")
    f.write(f"contamination: {best_params['contamination']}\n")
    f.write(f"\nМЕТРИКИ\n")
    f.write(f"Recall: {best_metrics['recall']:.3f}\n")
    f.write(f"Precision: {best_metrics['precision']:.3f}\n")
    f.write(f"F1: {best_metrics['f1']:.3f}\n")
    f.write(f"\nИспользованные признаки: {FEATURE_COLS}\n")

print("Результаты сохранены в 'anomaly_detection_results.txt'")
print("\n✅ Готово! Модель обучена, протестирована и сохранена.")