LABELS_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"
FEATURES_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
FINDING_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/finding_label.csv"
DISEASE_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv"

## 划分数据集
    Train / Val / Test（ 0.7 / 0.1 / 0.2）
    把三个 ID 列表落盘保存（JSON/CSV 都行），以后任何实验都复用这三个列表

## model 1训练:
    标签使用 finding_label.csv （在ptbxl_statements.csv取得映射）
    特征使用 12sl_features.csv的全部特征

## model 2训练：
    标签使用 disease_label.csv（在ptbxl_statements.csv取得映射）
    特征使用 12sl_features.csv的全部特征 + model1的output
    对所有 Train 样本计算 finding loss
    仅对 disease 有标注的 Train 样本计算 disease loss（缺失样本 disease loss mask 掉）

### 特征合并具体实现

model1 的 output 与 12sl_features.csv 的特征通过以下方式合并：

1. **Train 集：使用 OOF (Out-of-Fold) 预测**
   - 使用 5-fold MultilabelStratifiedKFold 对训练集进行划分
   - 每个 fold 中：用 4/5 训练数据训练 model1，预测剩余 1/5 的 finding 概率
   - 最终得到 Train 集的 OOF 预测 `Ptr_oof`，形状为 (N_train, Lf)，其中 Lf 是 finding 标签数
   - 合并方式：`Ztr = np.concatenate([Xtr, Ptr_oof], axis=1)`
   - 这样可以避免模型1的预测结果过拟合到训练集

2. **Val/Test 集：使用全量训练的 model1 预测**
   - 在生成 OOF 预测后，使用全部训练集重新训练一个 model1 (`model1_final`)
   - 用 `model1_final` 对 Val 和 Test 集进行预测，得到 `Pva` 和 `Pte`
   - 合并方式：
     - `Zva = np.concatenate([Xva, Pva], axis=1)`
     - `Zte = np.concatenate([Xte, Pte], axis=1)`

3. **最终特征维度**
   - 假设 12sl_features.csv 有 F 个特征，finding 有 Lf 个标签
   - 合并后的特征维度 Z = F + Lf
   - 例如：12sl_features.csv 有 100 个特征，finding 有 15 个标签，则 model2 的输入维度为 115

## 评估

### model 1
    model_1 是标准多标签抽取任务，用：
    - Micro-F1 / Macro-F1 / Hamming Loss / per-finding F1
    - finding-level recall：findings（例如 ST 段抬高/压低、T 波倒置、Q 波等）的召回率
    会更有说服力。

### 下游评估的时候也要mask掉测试集里不包含disease label的数据





