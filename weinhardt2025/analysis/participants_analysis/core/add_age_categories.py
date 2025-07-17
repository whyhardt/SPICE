import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCNinfo_Share.csv'
df = pd.read_csv(file_path)

# Original category conditions
conditions = [
    (df['ID'] >= 1) & (df['ID'] <= 221),
    (df['ID'] >= 300) & (df['ID'] <= 365),
    (df['ID'] > 400) & (df['age - years'] < 23)
]
choices = [1, 2, 3]
df['Category'] = np.select(conditions, choices, default=0)
df['Category'] = df['Category'].replace({2: 3, 3: 2})

df['age_floored'] = df['age - years'].apply(lambda x: int(np.floor(x)) if pd.notna(x) else np.nan)

# Add category from Maria's paper based on age bins using floored ages
age_conditions = [
    (df['age_floored'] >= 8) & (df['age_floored'] <= 10),   # 8-10yo
    (df['age_floored'] >= 11) & (df['age_floored'] <= 13),  # 11-13yo
    (df['age_floored'] >= 14) & (df['age_floored'] <= 15),  # 14-15yo
    (df['age_floored'] >= 16) & (df['age_floored'] <= 17),  # 16-17yo
    (df['age_floored'] >= 18) & (df['age_floored'] <= 24),  # 18-24yo
    (df['age_floored'] >= 25) & (df['age_floored'] <= 30)   # 25-30yo
]
age_choices = [1, 2, 3, 4, 5, 6]
df['age_maria'] = np.select(age_conditions, age_choices, default=0)

# Filter out uncategorized subjects
df_filtered = df[df['Category'] != 0].copy()

output_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCN.csv'
df_filtered.to_csv(output_path, index=False)


output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots'
os.makedirs(output_dir, exist_ok=True)


# 3 categories plot
colors = {1: 'red', 2: 'blue', 3: 'green'}
plt.figure(figsize=(12, 8))
for category in [1, 2, 3]:
    mask = df_filtered['Category'] == category
    if mask.any():
        plt.scatter(
            df_filtered[mask]['ID'],
            df_filtered[mask]['age - years'],
            c=colors[category],
            label=f'Category {category}',
            alpha=0.7,
            s=30
        )
plt.xlabel('ID')
plt.ylabel('Age (years)')
plt.title('Subject ID vs Age Category')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_file = os.path.join(output_dir, 'id_vs_age_3cat_plot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

# 6 categories plot
age_maria_colors = {1: 'purple', 2: 'orange', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'brown'}
age_maria_labels = {1: '8-10yo', 2: '11-13yo', 3: '14-15yo', 4: '16-17yo', 5: '18-24yo', 6: '25-30yo'}

plt.figure(figsize=(12, 8))
for age_cat in [1, 2, 3, 4, 5, 6]:
    mask = (df_filtered['age_maria'] == age_cat)
    if mask.any():
        plt.scatter(
            df_filtered[mask]['ID'],
            df_filtered[mask]['age - years'],
            c=age_maria_colors[age_cat],
            label=f'Age  {age_cat}: {age_maria_labels[age_cat]}',
            alpha=0.7,
            s=30
        )
plt.xlabel('ID')
plt.ylabel('Age (years)')
plt.title('Subject ID vs Age Categories')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
age_maria_output_file = os.path.join(output_dir, 'id_vs_age_6cat_plot.png')
plt.savefig(age_maria_output_file, dpi=300, bbox_inches='tight')
plt.close()