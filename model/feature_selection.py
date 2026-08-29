"""
特征筛选代码 - 使用XGBoost筛选特征
标签集合使用 finding_label.csv + disease_label.csv
筛选得到的特征结构与 selected_features.csv 一致
"""
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ================= 路径配置 =================
LABELS_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"
FEATURES_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
FINDING_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/finding_label.csv"
DISEASE_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv"
OUTPUT_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/selected_features.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_LABEL_FREQUENCY = 50

# ================= 数据加载 =================
def load_data():
    """加载特征与原始标签数据"""
    print("=" * 60)
    print("加载特征与标签数据...")
    df_labels = pd.read_csv(LABELS_PATH).set_index("ecg_id")
    df_features = pd.read_csv(FEATURES_PATH).set_index("ecg_id")

    common_ids = df_labels.index.intersection(df_features.index)
    df_labels = df_labels.loc[common_ids]
    df_features = df_features.loc[common_ids]

    print(f"对齐样本数: {len(common_ids)}")
    return df_features, df_labels

def load_label_mapping(label_path):
    """加载标签映射: SNOMED ID -> 标签名称"""
    df = pd.read_csv(label_path)
    mapping = dict(zip(df['snomed_id'], df['label']))
    label_type = "finding" if "finding" in label_path else "disease"
    print(f"加载{label_type}标签映射: {len(mapping)} 个标签定义")
    return mapping

# ================= 标签处理 =================
def process_labels(df_labels, label_mapping, min_freq=50):
    """处理标签,转换为二值化矩阵"""
    if 'scp_codes_ext_snomed' not in df_labels.columns:
        print("警告: 数据中未找到 'scp_codes_ext_snomed' 列")
        return None
    
    y_raw = df_labels['scp_codes_ext_snomed'].apply(
        lambda x: [label_mapping.get(item[0]) for item in ast.literal_eval(x) 
                  if item[0] in label_mapping]
    )
    
    y_raw = y_raw.apply(lambda x: [l for l in x if l is not None])
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_matrix = mlb.fit_transform(y_raw)
    y_df = pd.DataFrame(y_matrix, columns=mlb.classes_, index=df_labels.index)
    
    label_counts = y_df.sum(axis=0)
    keep_labels = label_counts[label_counts >= min_freq].index
    
    print(f"  初始标签数: {len(mlb.classes_)}")
    print(f"  保留标签数 (freq >= {min_freq}): {len(keep_labels)}")
    
    return y_df[keep_labels]

# ================= 特征筛选 =================
def select_features_with_xgboost(X, y, n_features=50):
    """
    使用XGBoost进行特征筛选
    对每个标签训练XGBoost模型,汇总特征重要性
    """
    print("\n" + "=" * 60)
    print(f"使用XGBoost进行特征筛选 (目标特征数: {n_features})")
    print("=" * 60)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 处理缺失值
    X_train_filled = X_train.fillna(X_train.median())
    X_val_filled = X_val.fillna(X_train.median())
    
    # 存储所有标签的特征重要性
    feature_importance_dict = defaultdict(list)
    
    # 对每个标签训练XGBoost模型
    print(f"\n对 {y_train.shape[1]} 个标签分别训练XGBoost模型...")
    processed_count = 0
    for i, label in enumerate(y_train.columns):
        y_train_label = y_train.iloc[:, i]
        
        # 跳过样本数过少的标签
        if y_train_label.sum() < MIN_LABEL_FREQUENCY:
            continue
        
        processed_count += 1
        if processed_count % 5 == 0:
            print(f"  已处理 {processed_count} 个有效标签...")
        
        # 训练XGBoost模型
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            verbosity=0
        )
        
        try:
            model.fit(X_train_filled, y_train_label)
            # 获取特征重要性
            importance = model.feature_importances_
            for j, feat in enumerate(X_train.columns):
                feature_importance_dict[feat].append(importance[j])
        except Exception as e:
            continue
    
    print(f"  共处理 {processed_count} 个有效标签")
    
    # 计算每个特征的平均重要性
    print("\n计算特征平均重要性...")
    avg_importance = {}
    for feat, importances in feature_importance_dict.items():
        avg_importance[feat] = np.mean(importances)
    
    # 排序并选择top n特征
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:n_features]
    
    # 创建DataFrame保存结果
    result_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    
    print(f"\n特征筛选完成!")
    print(f"  总特征数: {len(X.columns)}")
    print(f"  选中特征数: {len(top_features)}")
    print(f"\nTop 10 特征:")
    for i, (feat, imp) in enumerate(top_features[:10]):
        print(f"  {i+1:2d}. {feat:<40} {imp:.6f}")
    
    return result_df

# ================= 主流程 =================
def main():
    print("\n" + "=" * 60)
    print("XGBoost特征筛选")
    print("=" * 60)
    
    # 1. 加载数据
    X_raw, df_labels = load_data()
    
    # 2. 加载标签映射
    finding_mapping = load_label_mapping(FINDING_LABEL_PATH)
    disease_mapping = load_label_mapping(DISEASE_LABEL_PATH)
    
    # 3. 处理标签 - 合并finding和disease标签
    print("\n" + "=" * 60)
    print("处理 finding 标签...")
    y_finding = process_labels(df_labels, finding_mapping, MIN_LABEL_FREQUENCY)
    
    print("\n" + "=" * 60)
    print("处理 disease 标签...")
    y_disease = process_labels(df_labels, disease_mapping, MIN_LABEL_FREQUENCY)
    
    if y_finding is None or len(y_finding.columns) == 0:
        print("错误: finding 标签处理失败或没有有效标签")
        return
    
    if y_disease is None or len(y_disease.columns) == 0:
        print("错误: disease 标签处理失败或没有有效标签")
        return
    
    # 4. 合并标签
    print("\n" + "=" * 60)
    print("合并标签...")
    y_combined = pd.concat([y_finding, y_disease], axis=1)
    print(f"  Finding 标签数: {len(y_finding.columns)}")
    print(f"  Disease 标签数: {len(y_disease.columns)}")
    print(f"  合并后标签数: {len(y_combined.columns)}")
    
    # 5. 对齐特征和标签
    common_ids = X_raw.index.intersection(y_combined.index)
    X = X_raw.loc[common_ids]
    y = y_combined.loc[common_ids]
    
    print(f"\n最终数据集:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(X.columns)}")
    print(f"  标签数: {len(y.columns)}")
    
    # 6. 特征筛选
    selected_features_df = select_features_with_xgboost(X, y, n_features=50)
    
    # 7. 保存结果
    selected_features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n筛选结果已保存到: {OUTPUT_PATH}")
    
    print("\n特征筛选流程完成！")

if __name__ == "__main__":
    main()