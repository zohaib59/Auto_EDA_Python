import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shutil
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12})

# Load your dataset
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("zomato.csv", encoding='Latin1')
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Output folder setup (overwrite every time)
output_folder = "eda_outputs"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Universal EDA Function
# -------------------------
def auto_eda(data, target_column=None, sample_limit=1000):
    print(f"✅ Dataset Loaded | Shape: {data.shape}")
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if target_column:
        if target_column in cat_cols: cat_cols.remove(target_column)
        if target_column in num_cols: num_cols.remove(target_column)

    print(f"\n🧾 Categorical columns: {cat_cols}")
    print(f"🧮 Numerical columns: {num_cols}")

    # Sample for speed
    plot_data = data.copy()
    if len(data) > sample_limit:
        plot_data = data.sample(sample_limit, random_state=42)

    neon_palette = sns.color_palette("hsv", len(plot_data))

    # Dynamic Top Categories (City, Cuisine, etc.)
    top_candidates = ['city', 'resturant', 'cuisine', 'item', 'seller']
    for name in top_candidates:
        matching_cols = [col for col in cat_cols if name.lower() in col.lower()]
        for col in matching_cols:
            plt.figure(figsize=(10, 4))
            top_vals = plot_data[col].value_counts().nlargest(10)
            sns.barplot(x=top_vals.index, y=top_vals.values, palette="coolwarm")
            plt.title(f"Top 10 {col}", fontsize=14, fontweight='bold')
            plt.ylabel("Count"); plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"top10_{col}.png"))
            plt.close()

    # Countplots for other categoricals
    for col in cat_cols:
        if plot_data[col].nunique() <= 30:
            plt.figure(figsize=(10, 4))
            ax = sns.countplot(x=col, data=plot_data, palette=neon_palette, order=plot_data[col].value_counts().index)
            ax.set_title(f"{col}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col); ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            for container in ax.containers:
                ax.bar_label(container, label_type='edge', padding=2, fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"countplot_{col}.png"))
            plt.close()

    # Histograms for numericals
    for col in num_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(plot_data[col], bins=25, kde=True, color='cyan')
        plt.title(f"{col} Distribution", fontsize=14, fontweight='bold')
        plt.xlabel(col); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"histogram_{col}.png"))
        plt.close()

    # Pie Charts (top 3)
    for col in cat_cols[:3]:
        plt.figure(figsize=(6, 6))
        plot_data[col].value_counts().nlargest(6).plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3.colors,
            textprops={'fontsize': 11})
        plt.title(f"{col} Distribution", fontsize=14, fontweight='bold')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"pie_{col}.png"))
        plt.close()

    # Correlation heatmap
    if len(num_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = plot_data[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='cubehelix', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()

    # Pairplot
    if len(num_cols) >= 3:
        pair = sns.pairplot(plot_data[num_cols], diag_kind='kde', corner=True, palette='husl')
        pair.fig.suptitle("Numerical Feature Relationships", fontsize=16, fontweight='bold', y=1.02)
        pair.savefig(os.path.join(output_folder, "pairplot.png"))
        plt.close()

    # 3D Scatter
    if len(num_cols) >= 3:
        fig = px.scatter_3d(plot_data,
                            x=num_cols[0],
                            y=num_cols[1],
                            z=num_cols[2],
                            color=target_column if target_column else None,
                            title="3D Scatter Plot",
                            template="plotly_dark")
        fig.update_traces(marker=dict(size=4))
        fig.write_image(os.path.join(output_folder, "3d_scatter_plot.png"))

    # Bubble Plot
    if len(num_cols) >= 2:
        fig = px.scatter(plot_data,
                         x=num_cols[0],
                         y=num_cols[1],
                         size=num_cols[1],
                         color=target_column if target_column else None,
                         title="Bubble Plot",
                         template="plotly_dark")
        fig.write_image(os.path.join(output_folder, "bubble_plot.png"))

# Call the function with your dataset
auto_eda(data, target_column='Dining_Rating')  # Change target_column as needed
