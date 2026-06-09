import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


class DataPreprocessor:
    def __init__(self, window_size: int = 60, step: int = 30):
        """
        window_size: сколько строк в одном окне (например, 60 = 1 минута, если данные с частотой 1 Гц)
        step: шаг сдвига окна (например, 30 = окна пересекаются)
        """
        self.window_size = window_size
        self.step = step

    def load_and_filter(self, file_path: Path, allowed_classes: List[int]) -> pd.DataFrame:
        """Загружает parquet и оставляет только разрешенные классы."""
        df = pd.read_parquet(file_path)
        # Преобразуем индекс в столбец, если это timestamp
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df.index)
        else:
            df = df.reset_index().rename(columns={'index': 'timestamp'})

        # Фильтруем по классам
        df_filtered = df[df['class'].isin(allowed_classes)].copy()
        print(f"Файл {file_path.name}: {len(df_filtered)} строк после фильтрации (было {len(df)}).")
        return df_filtered

    def create_windows(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Нарезает DataFrame на окна для обучения/предсказания.

        Возвращает:
            windows: numpy array формы (n_windows, window_size, n_features)
        """
        data = df[feature_columns].values
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.step):
            window = data[i:i + self.window_size]
            windows.append(window)
        return np.array(windows)

    def prepare_normal_data(self, file_paths: List[Path], feature_columns: List[str]) -> np.ndarray:
        """Загружает несколько файлов, фильтрует только class==0 и создает окна."""
        all_windows = []
        for fpath in file_paths:
            df_normal = self.load_and_filter(fpath, allowed_classes=[0])
            if len(df_normal) > self.window_size:
                windows = self.create_windows(df_normal, feature_columns)
                all_windows.append(windows)
                print(f"  Создано {windows.shape[0]} окон из {fpath.name}")

        if not all_windows:
            raise ValueError("Не удалось создать ни одного окна для нормальных данных!")
        return np.vstack(all_windows)
