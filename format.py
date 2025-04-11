import pandas as pd

# 기존 파일 경로
file_path = 'C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\results\\adam_training_results.csv'

# CSV 불러오기
df = pd.read_csv(file_path)

# 'optimizer_type' 열을 제일 앞에 삽입
df.insert(0, 'optimizer_type', 4)  # 0은 optimizer_type 값

filtered_df = df[
    (df['norm_type'] == 0) &
    (df['learning_rate'].isin([0.01, 0.001])) &
    (df['kernel_num'].isin([4, 8])) &
    (df['init_type'] == 1)
]

# 수정된 CSV 저장
filtered_df.to_csv(file_path, index=False)

print(f"'optimizer_type' 열 추가 완료: {file_path}")
