import pandas as pd
import matplotlib.pyplot as plt
import os

# 설정
output_dir = 'C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\results'
csv_path = os.path.join(output_dir, 'optimizers_training_results.csv')

# 데이터 불러오기
df = pd.read_csv(csv_path)

# init_type=1로 필터
df = df[df['init_type'] == 1]

# 사용할 고정값들
learning_rates = [0.01, 0.001]
kernel_nums = [4, 8]
optimizer_labels = {
    0: 'SGD',
    1: 'Momentum',
    2: 'Adagrad',
    3: 'RMSprop',
    4: 'Adam'
}

# subplot 구성 (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training Loss by Optimizer\n(init_type = 1)', fontsize=16)

# 순회하며 각 칸 채우기
for i, lr in enumerate(learning_rates):
    for j, kernel in enumerate(kernel_nums):
        ax = axes[i][j]
        subset = df[(df['learning_rate'] == lr) & (df['kernel_num'] == kernel)]
        
        for opt in sorted(subset['optimizer_type'].unique()):
            opt_data = subset[subset['optimizer_type'] == opt]
            ax.plot(opt_data['epoch'], opt_data['avg_loss'], label=optimizer_labels.get(opt, f'Opt {opt}'))
        
        ax.set_title(f'LR={lr}, Kernel={kernel}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Loss')
        ax.grid(True)
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 저장
output_path = os.path.join(output_dir, 'optimizer_comparison_by_lr_kernel.png')
plt.savefig(output_path)
plt.close()

print(f"Graph saved to {output_path}")
