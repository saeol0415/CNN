
import pandas as pd
import matplotlib.pyplot as plt
import os

# 출력 디렉토리 설정 및 확인
output_dir = 'C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\results'

# CSV 파일 읽기
csv_path = os.path.join(output_dir, 'adam_training_results.csv')
df = pd.read_csv(csv_path, skiprows=0)  # 첫 번째 줄 건너뛰기 추가

# 전체 데이터에 대해 한 번에 그래프 그리기
plt.figure(figsize=(15, 15))
plt.suptitle('Training Results (Adam Optimizer)')

# 커널 개수와 초기화 방법에 따른 서브플롯 생성 (3x3)
kernel_nums = sorted(df['kernel_num'].unique())
init_types = sorted(df['init_type'].unique())

for i, init in enumerate(init_types):
    for j, kernel in enumerate(kernel_nums):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.title(f'Kernel={kernel}, Init={init}')

        # 해당 조건에 맞는 데이터 필터링
        subset = df[(df['kernel_num'] == kernel) & (df['init_type'] == init)]

        # 학습률과 정규화 타입에 따라 그래프 그리기
        for norm in subset['norm_type'].unique():
            for lr in subset['learning_rate'].unique():
                data = subset[(subset['norm_type'] == norm) & (subset['learning_rate'] == lr)]
                plt.plot(data['epoch'], data['avg_loss'], label=f'LR={lr}, Norm={norm}', marker=None)

        # 추가 메트릭 정보 출력 (선택 사항, 마지막 에폭 데이터 사용)
        last_row = subset.iloc[-1]  # 마지막 에폭 데이터
        plt.text(0.02, 0.98, f'Total Time: {last_row["total_time"]:.2f}s\n'
                             f'Converged: {last_row["converged"]}\n'
                             f'Conv Epoch: {last_row["convergence_epoch"]}\n'
                             f'Accuracy: {last_row["accuracy"]:.4f}', 
                 transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')

        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend(framealpha=0.5, loc='upper right')
        plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_image = os.path.join(output_dir, 'adam_training_results_full.png')
plt.savefig(output_image)
plt.close()

print(f"Graph saved to {output_image}")