# 匯入必要套件
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from matplotlib import font_manager, rc
import csv
import time

# 設定字體以支援中文
font_path = "Base/MSGOTHIC.TTF"  
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())

# 自動尋找標籤欄位
# 根據常見的標籤欄位名稱或類別資料的特徵進行判斷
def find_label_column(df):
    """自動尋找標籤欄位"""
    # 常見的標籤欄位名稱
    possible_names = ['label', 'class', 'target', 'category', 'type']
    
    # 檢查所有欄位名稱
    for col in df.columns:
        # 將欄位名稱轉換為小寫並去除空白來比較
        col_clean = col.strip().lower()
        if col_clean in possible_names:
            return col
        
        # 檢查欄位是否包含類別資料（通常標籤欄位的唯一值數量較少）
        if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.1:
            return col
    
    raise ValueError("找不到適合的標籤欄位")

# 移除低方差特徵
# 使用 VarianceThreshold 過濾方差低於指定閾值的特徵
def remove_low_variance_features(X, threshold=0.01):
    """移除低方差特徵"""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    kept_features = selector.get_support()
    return X_filtered, kept_features

# 移除高度相關特徵
# 根據相關係數矩陣過濾相關性高於指定閾值的特徵
def remove_highly_correlated_features(X, threshold=0.95):
    """移除高度相關特徵"""
    # 計算相關係數矩陣
    corr_matrix = pd.DataFrame(X).corr().abs()
    
    # 獲取要移除的特徵索引
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # 保留的特徵索引
    kept_features = ~pd.DataFrame(X).columns.isin(to_drop)
    
    # 移除高度相關的特徵
    X_filtered = pd.DataFrame(X).drop(columns=to_drop).values
    
    return X_filtered, kept_features

# 使用 F-test 選擇最重要的特徵
def select_k_best_features(X, y, k='auto'):
    """使用 F-test 選擇最重要的特徵"""
    if k == 'auto':
        k = int(X.shape[1] * 0.7)  # 預設保留 70% 的特徵
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_filtered = selector.fit_transform(X, y)
    kept_features = selector.get_support()
    
    return X_filtered, kept_features

# 生成特徵重要性報告
def get_feature_importance_report(original_features, kept_features_mask, feature_names):
    """生成特徵重要性報告"""
    report = []
    for i, (original, kept) in enumerate(zip(original_features, kept_features_mask)):
        if not kept:
            report.append(f"移除特徵 '{feature_names[i]}'")
    return report

# 檢查 CSV 檔案的欄位名稱
def check_csv_columns(csv_path):
    # 檢查CSV欄位
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns
        return cols.tolist()
    except Exception as e:
        print(f"檢查CSV欄位時發生錯誤：{e}")
        return None

# 讀取 CSV 檔案並顯示基本資訊
def read_csv(csv_path, nrows=320000):
    # 讀取資料
    try:
        df = pd.read_csv(csv_path, nrows=nrows)
        print(f"成功讀取 {len(df)} 筆資料")
        # 只顯示簡短的資訊摘要
        print("\n資料基本資訊:")
        print(f"- 總列數: {len(df)}")
        print(f"- 總欄位數: {len(df.columns)}")
        print(f"- 記憶體使用: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        return df
    except Exception as e:
        print(f'讀取 CSV 發生錯誤: {e}')
        sys.exit(1)

# 定義 CNN-LSTM 模型類別
# 包含 CNN 層、LSTM 層和全連接層
class CNNLSTM(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CNNLSTM, self).__init__()
        # CNN 部分
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 計算LSTM的輸入特徵數
        self.lstm_input_size = 64 * (input_features // 4)  # 因為有兩次池化，所以特徵長度會減半兩次

        # LSTM 部分
        self.lstm = nn.LSTM(input_size=64,  # 使用CNN輸出的通道數作為輸入特徵
                          hidden_size=128, 
                          num_layers=2, 
                          batch_first=True, 
                          dropout=0.3,
                          bidirectional=True)

        # 自注意力機制
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # 全連接層
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)  # 256是因為雙向LSTM
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # CNN 層
        x = x.view(batch_size, 1, -1)  # 重塑為 (batch_size, channels, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 準備LSTM的輸入
        x = x.permute(0, 2, 1)  # 將形狀從 (batch, channels, seq_len) 變為 (batch, seq_len, channels)

        # LSTM 層
        x, _ = self.lstm(x)

        # 自注意力機制
        x, _ = self.attention(x, x, x)

        # 全連接層
        x = x[:, -1, :]  # 取最後一個時間步的輸出
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = torch.relu(x)

        x = self.fc3(x)
        return x

# 分離特徵與標籤
# 將資料集分為特徵矩陣 X 和標籤向量 y
def split_features_labels(df):
    # 找到 Label 欄位（考慮可能有前後空格）
    label_col = next(col for col in df.columns if col.strip() == 'Label')
    
    # 取得特徵欄位（去除 Label 欄位）
    feature_cols = [col for col in df.columns if col != label_col]
    
    X = df[feature_cols].values
    
    # 將類別標籤轉換為二分類：BENIGN為0（正常），其他都為1（異常）
    y = df[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1).values
    
    # 檢查標籤是否都有被正確轉換
    assert not np.any(pd.isna(y)), f'有未知的標籤類別！'
    return X, y

# 特徵標準化
# 將特徵縮放到標準正態分佈
def scale_features(X):
    # 將 inf/-inf 轉為 nan
    X = np.where(np.isfinite(X), X, np.nan)
    # 用每欄平均值填補 nan
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# 調整資料形狀以適應 CNN-LSTM 模型
def reshape_for_cnnlstm(X):
    return X.reshape((X.shape[0], 1, X.shape[1]))

# 分割訓練集與測試集
def split_train_test(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 建立 DataLoader
# 將 numpy array 轉為 PyTorch Tensor 並處理資料不平衡問題
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 建立張量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 處理資料不平衡
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    samples_weights = weights[y_train]
    samples_weights = torch.from_numpy(samples_weights)
    sampler = WeightedRandomSampler(weights=samples_weights,
                                   num_samples=len(samples_weights),
                                   replacement=True)
    
    # 建立資料集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 建立資料載入器，訓練集使用 sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, device

# 執行一個訓練 epoch
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """執行一個訓練 epoch
    
    Returns:
        tuple: (train_loss, train_preds, train_labels)
    """
    model.train()
    running_loss = 0.0
    train_preds = []
    train_labels = []
    
    pbar = tqdm(train_loader, desc='訓練中', leave=True)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(train_loader.dataset), train_preds, train_labels

# 執行一個驗證 epoch
def validate_epoch(model, test_loader, criterion, device):
    """執行一個驗證 epoch
    
    Returns:
        tuple: (val_loss, val_preds, val_labels)
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='驗證中'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    return val_loss / len(test_loader.dataset), val_preds, val_labels

# 訓練模型的主函數
# 包含早停機制和最佳模型保存
def train_model(model, train_loader, test_loader, device, epochs=32, lr=0.00002, patience=5):
    """訓練模型的主函數"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'accuracy': [], 'train_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 
        'gpu_mem_MB': []
    }
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_f1 = 0
    patience_counter = 0
    best_model_path = 'Model/CNN_LSTM.pth'

    for epoch in range(epochs):
        # 訓練階段

        torch.cuda.reset_peak_memory_stats(0)
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        gpu_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0

        # 驗證階段
        val_loss, val_preds, val_labels = validate_epoch(
            model, test_loader, criterion, device
        )
        
        # 計算訓練和驗證指標
        train_metrics = calculate_metrics(train_labels, train_preds)
        val_metrics = calculate_metrics(val_labels, val_preds)
        
        # 更新歷史記錄
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics["accuracy"])
        history['accuracy'].append(val_metrics['accuracy'])
        history['f1'].append(val_metrics['f1'])  # 確保記錄 F1 指標
        history['precision'].append(val_metrics['precision'])  # 確保記錄精確率
        history['recall'].append(val_metrics['recall'])  # 確保記錄召回率
        history['gpu_mem_MB'].append(gpu_mem)
        
        # 輸出當前效能指標
        print(f'\nEpoch {epoch+1} 結果:')
        print(f'訓練 Loss: {train_loss:.4f}')
        print(f'訓練準確率: {train_metrics["accuracy"]:.4f}')
        print(f'-------------------')
        print(f'驗證 Loss: {val_loss:.4f}')
        print(f'驗證準確率: {val_metrics["accuracy"]:.4f}')
        print(f'GPU 記憶體: {gpu_mem} MB')
        
        # 早停機制
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'保存新的最佳模型 (F1: {val_metrics["f1"]:.4f})')
        else:
            patience_counter += 1
            print(f'沒有改善，耐心值: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'\n早停：{patience} 個 epoch 未改善')
                break
    
    # 載入最佳模型
    print(f'\n載入最佳模型 (F1: {best_f1:.4f})')
    model.load_state_dict(torch.load(best_model_path))
    return history, model

# 測試模型並回傳預測結果與正確標籤
def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# 計算測試準確率
def compute_accuracy(all_labels, all_preds):
    accuracy = (all_labels == all_preds).mean()
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# 繪製訓練與驗證損失曲線
def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(10, 6))
    # 繪製損失曲線
    ax.plot(history['epoch'], history['train_loss'], label='訓練 Loss')
    ax.plot(history['epoch'], history['val_loss'], label='驗證 Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('訓練與驗證損失曲線')
    ax.legend()
    # 儲存圖片
    plt.savefig('Image/訓練與驗證損失曲線.png')
    plt.close()

def plot_accuracy_curve(history):
    """繪製訓練與驗證的準確度曲線"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 繪製準確度曲線
    ax.plot(history['epoch'], history['accuracy'], label='驗證準確度')
    ax.plot(history['epoch'], history.get('train_accuracy', [0]*len(history['epoch'])), label='訓練準確度')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('訓練與驗證準確度曲線')
    ax.legend()

    # 儲存圖片
    plt.savefig('Image/訓練與驗證準確度曲線.png')
    plt.close()

# 繪製 GPU 記憶體使用情況曲線
def plot_gpu_memory(history):
    plt.figure()
    plt.plot(history['epoch'], history['gpu_mem_MB'], label='GPU Memory (MB)')
    plt.xlabel('Epoch')
    plt.ylabel('GPU Memory (MB)')
    plt.title('GPU 記憶體使用情況')
    plt.savefig('Image/GPU記憶體使用情況.png')

# 繪製混淆矩陣
def plot_confusion_matrix(all_labels, all_preds):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('混淆矩陣')
    plt.savefig('Image/混淆矩陣.png')

# 繪製 ROC 曲線 (僅限二分類)
def plot_roc_curve(model, test_loader, device, all_labels):
    from sklearn.metrics import roc_curve, auc
    if len(np.unique(all_labels)) == 2:
        y_score = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                y_score.extend(torch.softmax(outputs, 1)[:,1].cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('接收者操作特徵曲線')
        plt.legend(loc='lower right')
        plt.savefig('Image/接收者操作特徵曲線.png')

# 預處理資料
# 包括處理極端值與遺漏值
def preprocess_dataframe(df):
    # 複製一份資料以免修改到原始資料
    df = df.copy()
    
    # 對每個數值欄位進行處理
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int']).columns
    for col in numeric_cols:
        # 計算上下四分位數
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 設定極端值邊界
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # 將極端值替換為邊界值
        df[col] = df[col].clip(lower=lower, upper=upper)
        
        # 使用中位數填補遺漏值
        df[col] = df[col].fillna(df[col].median())
    
    return df

# 計算多個評估指標
def calculate_metrics(y_true, y_pred):
    """計算多個評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
    
    Returns:
        dict: 包含 accuracy, f1, precision, recall, roc_auc, mcc 的字典
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

    # 確保處理多分類問題時使用 'macro' 平均方式
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_true, y_pred, multi_class='ovr') if len(np.unique(y_true)) > 2 else roc_auc_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

# 繪製各種評估指標（F1、準確率、精確率、召回率）的曲線
def plot_evaluation_metrics(history):
    """繪製各種評估指標（F1、準確率、精確率、召回率）的曲線"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 繪製各種評估指標曲線
    ax.plot(history['epoch'], history['f1'], label='F1 Score')
    ax.plot(history['epoch'], history['accuracy'], label='Accuracy')
    ax.plot(history['epoch'], history['precision'], label='Precision')
    ax.plot(history['epoch'], history['recall'], label='Recall')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('模型評估指標曲線')
    ax.legend()

    # 儲存圖片
    plt.savefig('Image/模型評估指標曲線.png')
    plt.close()

# 將訓練過程中的 history 資料儲存為 CSV 檔案
def save_history_to_csv(history, file_path='history.csv'):
    """將訓練過程中的 history 資料儲存為 CSV 檔案"""
    keys = history.keys()
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for i in range(len(history['epoch'])):
            row = {key: history[key][i] for key in keys}
            writer.writerow(row)

def count_model_parameters(model):
    """計算模型的參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 主流程
if __name__ == '__main__':
    # 首先檢查CSV檔案的欄位
    print("檢查CSV檔案欄位...")
    data_path = 'Data/AllMerged.csv'
    actual_columns = check_csv_columns(data_path)
    if actual_columns is None:
        print("無法讀取CSV檔案，程式終止")
        sys.exit(1)
    
    # 讀取資料
    df = read_csv(data_path, nrows=10000000)
    
    # 預處理
    df = preprocess_dataframe(df)
    
    # 找到標籤欄位
    try:
        label_col = find_label_column(df)
        print(f'找到標籤欄位: {label_col}')
    except ValueError as e:
        print(f"錯誤：{e}")
        sys.exit(1)
    
    print('標籤分布：', df[label_col].value_counts())
    
    # 分離特徵與標籤
    X, y = split_features_labels(df)
    feature_names = [col for col in df.columns if col != label_col]
    
    print(f'\n開始特徵選擇...')
    print(f'原始特徵數量: {X.shape[1]}')
    
    # 1. 移除低方差特徵
    X_filtered, var_mask = remove_low_variance_features(X, threshold=0.002)
    print(f'\n移除低方差特徵後的特徵數量: {X_filtered.shape[1]}')
    var_report = get_feature_importance_report(X, var_mask, feature_names)
    print('\n低方差特徵移除報告:')
    for line in var_report:
        print(line)
    
    # 2. 移除高度相關特徵
    X_filtered, corr_mask = remove_highly_correlated_features(X_filtered, threshold=0.98)
    print(f'\n移除高度相關特徵後的特徵數量: {X_filtered.shape[1]}')
    corr_report = get_feature_importance_report(X, corr_mask, feature_names)
    print('\n高度相關特徵移除報告:')
    for line in corr_report:
        print(line)
    
    # 3. 選擇最重要的特徵
    X_filtered, importance_mask = select_k_best_features(X_filtered, y, k='auto')
    print(f'\n特徵選擇後的最終特徵數量: {X_filtered.shape[1]}')
    importance_report = get_feature_importance_report(X, importance_mask, feature_names)
    print('\n特徵重要性報告:')
    for line in importance_report:
        print(line)
    
    # 特徵標準化
    X = scale_features(X_filtered)
    
    # 調整資料形狀
    X = reshape_for_cnnlstm(X)
    
    # 分割訓練測試集
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # 建立資料載入器
    train_loader, test_loader, device = create_dataloaders(X_train, X_test, y_train, y_test)
    
    # 建立模型
    num_classes = len(np.unique(y))
    input_features = X.shape[2]
    model = CNNLSTM(input_features, num_classes).to(device)
    
    # 記錄開始時間
    start_time = time.time()

    # 訓練模型
    print('\n開始訓練模型...')
    history, model = train_model(model, train_loader, test_loader, device)

    # 記錄結束時間
    end_time = time.time()
    training_time = end_time - start_time

    # 計算模型參數數量
    model_parameters = count_model_parameters(model)

    # 將訓練時間和參數數量加入到歷史記錄中
    history['training_time'] = [training_time] * len(history['epoch'])
    history['model_parameters'] = [model_parameters] * len(history['epoch'])

    print(f'訓練時間: {training_time:.2f} 秒')
    print(f'模型參數數量: {model_parameters}')

    # 測試模型
    print('\n測試模型效能...')
    all_labels, all_preds = test_model(model, test_loader, device)
    compute_accuracy(all_labels, all_preds)
    
    # 繪製圖表
    print('\n生成圖片中...')
    plot_loss_curve(history)
    plot_accuracy_curve(history)
    plot_gpu_memory(history)
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(model, test_loader, device, all_labels)
    plot_evaluation_metrics(history)  
    print('圖片生成完成！')

    # 儲存 history 為 CSV
    save_history_to_csv(history, file_path='Image/history.csv')
    print('訓練歷史已儲存為 CSV 檔案！')