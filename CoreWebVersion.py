import matplotlib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import io

matplotlib.use('Agg')

def detection_methods(row):
    methods = []

    if row.get('isolation_forest_anomaly_flag', 0) == 1:
        methods.append('isolation_forest')

    if row.get('zscore_anomaly_flag', 0) == 1:
        methods.append('zscore')

    if row.get('dbscan_anomaly_flag', 0) == 1:
        methods.append('dbscan')

    return ','.join(methods)


def _save_plot_and_add_to_list(figure, plot_list_to_append):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = buf.read()
    plot_list_to_append.append(img_bytes)
    plt.close(figure)


def check_isolation_forest(df_input, contamination=0.05, visualize=False, plot_list_to_append=None):
    features = df_input.select_dtypes(include=[np.number]).columns.tolist()

    if not features:
        print("Isolation Forest: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    df_features_only = df_input[features].copy()
    df_features_only.dropna(inplace=True)

    if df_features_only.empty:
        print("Isolation Forest: Нет данных после удаления NaN из числовых признаков.")
        return pd.Series(0, index=df_features_only.index), 0.0

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions_raw = model.fit_predict(df_features_only)

    anomaly_flags_for_features = pd.Series(0, index=df_features_only.index)
    anomaly_flags_for_features[predictions_raw == -1] = 1

    final_anomaly_flags = pd.Series(0, index=df_input.index)
    final_anomaly_flags.update(anomaly_flags_for_features)

    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)

    print(f"Isolation Forest обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(features) >= 2 and plot_list_to_append is not None:
        plot_data = df_features_only.loc[anomaly_flags_for_features.index, features[:2]]
        if not plot_data.empty:
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                plot_data[features[0]],
                plot_data[features[1]],
                c=anomaly_flags_for_features,
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            plt.colorbar(label='Аномалия')
            plt.title('Результаты Isolation Forest')
            _save_plot_and_add_to_list(fig, plot_list_to_append)

    return final_anomaly_flags, percent


def check_dbscan(df_input, visualize=False, plot_list_to_append=None):
    features = df_input.select_dtypes(include=[np.number]).columns.tolist()

    if not features:
        print("DBSCAN: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    df_features_only = df_input[features].copy()
    df_features_only.dropna(inplace=True)

    if df_features_only.empty:
        print("DBSCAN: Нет данных после удаления NaN из числовых признаков.")
        return pd.Series(0, index=df_features_only.index), 0.0

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features_only)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)

    anomaly_flags_for_features = pd.Series(0, index=df_features_only.index)
    anomaly_flags_for_features[clusters == -1] = 1

    final_anomaly_flags = pd.Series(0, index=df_input.index)
    final_anomaly_flags.update(anomaly_flags_for_features)

    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)

    print(f"DBSCAN обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(features) >= 2 and plot_list_to_append is not None:
        plot_data = df_features_only.loc[anomaly_flags_for_features.index, features[:2]]
        if not plot_data.empty:
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                plot_data[features[0]],
                plot_data[features[1]],
                c=anomaly_flags_for_features,
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            plt.colorbar(label='Аномалия (DBSCAN)')
            plt.title('Результаты DBSCAN')
            _save_plot_and_add_to_list(fig, plot_list_to_append)

    return final_anomaly_flags, percent


def check_zscore(df_input, threshold=3.0, visualize=False, plot_list_to_append=None):
    numerical_columns = df_input.select_dtypes(include=[np.number]).columns.tolist()

    technical_cols = [
        'isolation_forest_anomaly_flag',
        'zscore_anomaly_flag',
        'dbscan_anomaly_flag',
        'is_warning',
    ]
    filtered_numerical_columns = [col for col in numerical_columns if col not in technical_cols and not col.endswith('_anomaly_flag')]

    if not filtered_numerical_columns:
        print("Z-score: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    overall_anomaly_flags_np = np.zeros(len(df_input), dtype=int)
    zscore_values_for_viz = pd.DataFrame(index=df_input.index)

    for col in filtered_numerical_columns:
        col_data = df_input[col].copy()
        col_mean = col_data.mean()
        col_std = col_data.std()

        if col_std == 0 or pd.isna(col_std) or pd.isna(col_mean):
            col_zscores = pd.Series(0.0, index=col_data.index)
        else:
            col_zscores = np.abs((col_data - col_mean) / col_std)

        zscore_values_for_viz[col + '_zscore'] = col_zscores.fillna(0)
        overall_anomaly_flags_np = np.where(
            col_zscores.fillna(0) > threshold,
            1,
            overall_anomaly_flags_np)

    final_anomaly_flags = pd.Series(overall_anomaly_flags_np, index=df_input.index)
    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)

    print(f"Z-score обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(filtered_numerical_columns) > 0 and plot_list_to_append is not None:
        if not zscore_values_for_viz.empty:
            fig = plt.figure(figsize=(12, min(8, 2 * len(filtered_numerical_columns))))
            cols_to_plot = min(4, len(filtered_numerical_columns))

            for i, col in enumerate(filtered_numerical_columns[:cols_to_plot]):
                plt.subplot( (cols_to_plot + 1) // 2, 2, i + 1)
                data_to_plot = zscore_values_for_viz[col + '_zscore'].replace([np.inf, -np.inf], np.nan).dropna()

                if not data_to_plot.empty:
                    sns.histplot(
                        data_to_plot, bins=30, kde=True
                    )
                    plt.axvline(
                        x=threshold,
                        color='r',
                        linestyle='--',
                        label=f'Порог ({threshold})'
                    )
                    plt.title(f'Распределение Z-score для {col}')
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center')
                    plt.title(f'Z-score для {col}')

            plt.tight_layout()
            _save_plot_and_add_to_list(fig, plot_list_to_append)

    return final_anomaly_flags, percent


def start(df, classification=False, visualize=False, plot_list_to_append=None):
    if plot_list_to_append is None:
        plot_list_to_append = []

    df_original_columns = df.columns.tolist()
    df_copy = df.copy()
    original_size = len(df_copy)
    df_copy.dropna(inplace=True)
    removed_rows = original_size - len(df_copy)

    if removed_rows > 0:
        print(f"При анализе не учитываются {removed_rows} строк с None/NaN значениями ({removed_rows/original_size*100:.2f}% от исходного датасета)\n")

    if df_copy.empty:
        print("После удаления строк с NaN значениями датасет стал пустым. Анализ невозможен.")
        columns_for_warning_csv = df_original_columns + ['detection_methods']
        warning_df_to_save = pd.DataFrame(columns=columns_for_warning_csv)
        treated_df_to_save = pd.DataFrame(columns=df_original_columns)
        print("Проверка завершена (нет данных для анализа).")
        return warning_df_to_save, treated_df_to_save, plot_list_to_append

    df_copy['isolation_forest_anomaly_flag'] = 0
    df_copy['zscore_anomaly_flag'] = 0
    df_copy['dbscan_anomaly_flag'] = 0

    isolation_flags, isolation_percent = check_isolation_forest(
        df_copy,
        visualize=visualize,
        plot_list_to_append=plot_list_to_append
    )
    df_copy.loc[isolation_flags.index, 'isolation_forest_anomaly_flag'] = isolation_flags

    zscore_flags, zscore_percent = check_zscore(
        df_copy,
        visualize=visualize,
        plot_list_to_append=plot_list_to_append
    )
    df_copy.loc[zscore_flags.index, 'zscore_anomaly_flag'] = zscore_flags

    dbscan_percent = 0.0
    if classification:
        dbscan_flags, dbscan_percent_val = check_dbscan(
            df_copy,
            visualize=visualize,
            plot_list_to_append=plot_list_to_append
        )
        df_copy.loc[dbscan_flags.index, 'dbscan_anomaly_flag'] = dbscan_flags
        dbscan_percent = dbscan_percent_val

    if classification:
        avg_pct = round((isolation_percent + dbscan_percent + zscore_percent) / 3, 2)
    else:
        avg_pct = round((isolation_percent + zscore_percent) / 2, 2) if isolation_percent + zscore_percent > 0 else 0.0

    print(f"Средний процент аномалий: {avg_pct}%")

    if avg_pct > 15:
        print(f"ВНИМАНИЕ: Обнаружены аномалии ({avg_pct}%). Рекомендуется проверить датасет.")
    else:
        print(f"Предположительно с датасетом все хорошо. Процент аномалий: {avg_pct}%")

    df_copy['is_warning'] = 0
    df_copy.loc[df_copy['isolation_forest_anomaly_flag'] == 1, 'is_warning'] = 1
    df_copy.loc[df_copy['zscore_anomaly_flag'] == 1, 'is_warning'] = 1
    if classification:
        df_copy.loc[df_copy['dbscan_anomaly_flag'] == 1, 'is_warning'] = 1

    warning_df = df_copy[df_copy['is_warning'] == 1].copy()
    columns_for_warning_csv = df_original_columns + ['detection_methods']

    if not warning_df.empty:
        warning_df['detection_methods'] = warning_df.apply(detection_methods, axis=1)
        cols_to_save_warning = [col for col in columns_for_warning_csv if col in warning_df.columns]
        warning_df_to_save = warning_df[cols_to_save_warning]
    else:
        warning_df_to_save = pd.DataFrame(columns=columns_for_warning_csv)

    treated_df = df_copy[df_copy['is_warning'] == 0].copy()
    cols_to_save_treated = [col for col in df_original_columns if col in treated_df.columns]
    treated_df_to_save = treated_df[cols_to_save_treated]

    print("Проверка завершена")

    return warning_df_to_save, treated_df_to_save, plot_list_to_append