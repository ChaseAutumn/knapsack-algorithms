import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持和字符显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'

def read_and_process_data(filename):
    """
    读取并处理实验结果CSV文件
    """
    try:
        # 尝试不同编码方式读取文件
        encodings = ['gbk', 'gb2312', 'utf-8', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filename, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("无法读取文件，请检查文件编码")
        
        # 显示数据基本信息
        print("数据基本信息:")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def clean_and_process_data(df):
    """
    清理和处理数据
    """
    df_clean = df.copy()
    
    # 处理执行时间列名
    time_col = None
    for col in df_clean.columns:
        if '执行时间' in col or 'time' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        print("未找到执行时间列")
        return None
    
    # 重命名列以便统一处理
    column_mapping = {time_col: '执行时间'}
    df_clean = df_clean.rename(columns=column_mapping)
    
    # 确保关键列存在
    required_columns = ['物品数量', '背包容量', '算法名称', '执行时间']
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    if missing_columns:
        print(f"缺少必要列: {missing_columns}")
        return None
    
    # 数据类型转换和清理
    try:
        # 转换数值列
        df_clean['物品数量'] = pd.to_numeric(df_clean['物品数量'], errors='coerce')
        df_clean['背包容量'] = pd.to_numeric(df_clean['背包容量'], errors='coerce')
        df_clean['执行时间'] = pd.to_numeric(df_clean['执行时间'], errors='coerce')
        
        # 移除无效数据
        df_clean = df_clean.dropna(subset=['物品数量', '背包容量', '执行时间'])
        df_clean = df_clean[df_clean['执行时间'] >= 0]  # 保留执行时间>=0的记录
        
        # 只保留物品数量<=200的数据
        df_clean = df_clean[df_clean['物品数量'] <= 200]
        print(f"筛选物品数量<=200后的数据形状: {df_clean.shape}")
        
        # 清理算法名称
        df_clean = clean_algorithm_names(df_clean)
        
        print(f"最终数据形状: {df_clean.shape}")
        print(f"算法类型: {list(df_clean['算法名称'].unique())}")
        print(f"物品数量范围: {df_clean['物品数量'].min()} - {df_clean['物品数量'].max()}")
        
        return df_clean
        
    except Exception as e:
        print(f"数据清理时出错: {e}")
        return None

def clean_algorithm_names(df):
    """
    清理算法名称，处理编码问题
    """
    # 创建算法名称映射
    algorithm_mapping = {}
    unique_algorithms = df['算法名称'].unique()
    
    for alg in unique_algorithms:
        alg_str = str(alg).strip()
        if '蛮' in alg_str or 'brute' in alg_str.lower() or '暴力' in alg_str:
            algorithm_mapping[alg] = '蛮力法'
        elif '动态' in alg_str or 'dynamic' in alg_str.lower() or 'dp' in alg_str.lower():
            algorithm_mapping[alg] = '动态规划法'  
        elif '贪' in alg_str or 'greedy' in alg_str.lower():
            algorithm_mapping[alg] = '贪心法'
        elif '回溯' in alg_str or 'backtrack' in alg_str.lower():
            algorithm_mapping[alg] = '回溯法'
        else:
            algorithm_mapping[alg] = alg_str
    
    df['算法名称'] = df['算法名称'].map(algorithm_mapping)
    return df

def create_execution_time_comparison(df, output_filename='algorithm_comparison.pdf'):
    """
    创建算法执行时间对比图表并保存为PDF
    """
    # 创建PDF文件
    with PdfPages(output_filename) as pdf:
        # 获取基本信息
        capacities = sorted(df['背包容量'].unique())
        algorithms = sorted(df['算法名称'].unique())
        
        # 设置颜色和标记
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        algorithm_style = {}
        for i, alg in enumerate(algorithms):
            algorithm_style[alg] = (colors[i % len(colors)], markers[i % len(markers)])
        
        # 图1: 整体执行时间对比（对数坐标）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('0-1背包问题算法执行时间对比分析（物品数量≤200）', fontsize=16, fontweight='bold')
        
        # 为每个背包容量创建子图
        for idx, capacity in enumerate(capacities[:4]):  # 最多显示4个容量
            if idx >= 4:
                break
                
            ax = axes[idx//2, idx%2]
            
            # 筛选当前容量的数据
            capacity_data = df[df['背包容量'] == capacity]
            
            for alg in algorithms:
                alg_data = capacity_data[capacity_data['算法名称'] == alg]
                if len(alg_data) > 0:
                    # 按物品数量排序并聚合相同物品数量的数据
                    alg_grouped = alg_data.groupby('物品数量')['执行时间'].mean().reset_index()
                    alg_grouped = alg_grouped.sort_values('物品数量')
                    
                    # 对于回溯法和蛮力法，过滤掉执行时间为0的数据点
                    if alg in ['回溯法', '蛮力法']:
                        alg_grouped = alg_grouped[alg_grouped['执行时间'] > 0]
                    
                    if len(alg_grouped) > 0:
                        color, marker = algorithm_style[alg]
                        ax.plot(alg_grouped['物品数量'], alg_grouped['执行时间'], 
                               label=alg, color=color, marker=marker, 
                               linewidth=2, markersize=6, alpha=0.8)
            
            ax.set_xlabel('物品数量', fontsize=12)
            ax.set_ylabel('执行时间 (ms)', fontsize=12)
            ax.set_title(f'背包容量: {capacity}', fontsize=12, fontweight='bold')
            ax.set_yscale('log')  # 使用对数坐标
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置x轴刻度
            if len(capacity_data) > 0:
                x_ticks = sorted(capacity_data['物品数量'].unique())
                if len(x_ticks) > 10:
                    # 如果刻度太多，只显示部分
                    step = max(1, len(x_ticks) // 8)
                    x_ticks = x_ticks[::step]
                ax.set_xticks(x_ticks)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 图2: 算法性能随规模变化趋势（线性坐标）
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 选择一个代表性的背包容量进行分析
        main_capacity = capacities[len(capacities)//2] if len(capacities) > 1 else capacities[0]
        main_data = df[df['背包容量'] == main_capacity]
        
        for alg in algorithms:
            alg_data = main_data[main_data['算法名称'] == alg]
            if len(alg_data) > 0:
                # 聚合相同物品数量的数据
                alg_grouped = alg_data.groupby('物品数量')['执行时间'].mean().reset_index()
                alg_grouped = alg_grouped.sort_values('物品数量')
                
                # 对于回溯法和蛮力法，过滤掉执行时间为0的数据点
                if alg in ['回溯法', '蛮力法']:
                    alg_grouped = alg_grouped[alg_grouped['执行时间'] > 0]
                
                if len(alg_grouped) > 0:
                    color, marker = algorithm_style[alg]
                    ax.plot(alg_grouped['物品数量'], alg_grouped['执行时间'], 
                           label=alg, color=color, marker=marker, 
                           linewidth=3, markersize=8, alpha=0.8)
        
        ax.set_xlabel('物品数量', fontsize=14)
        ax.set_ylabel('执行时间 (ms)', fontsize=14)
        ax.set_title(f'算法执行时间对比 (背包容量: {main_capacity})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # 添加性能分析文本
        textstr = '性能分析（物品数量≤200）:\n• 蛮力法: 指数时间复杂度O(2^n)\n• 动态规划法: 时间复杂度O(n×W)\n• 贪心法: 时间复杂度O(n log n)\n• 回溯法: 最坏情况O(2^n)\n\n注: 仅分析物品数量≤200的情况\n回溯法和蛮力法的零值已过滤'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 图3: 算法性能热力图
        if len(main_data) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建数据透视表
            pivot_data = main_data.groupby(['算法名称', '物品数量'])['执行时间'].mean().unstack(fill_value=0)
            
            if not pivot_data.empty:
                # 创建热力图
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': '执行时间 (ms)'})
                ax.set_title(f'算法执行时间热力图 (背包容量: {main_capacity})', fontsize=14, fontweight='bold')
                ax.set_xlabel('物品数量', fontsize=12)
                ax.set_ylabel('算法类型', fontsize=12)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
        
        # 图4: 算法效率比较（条形图）
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算平均执行时间
        avg_times = df.groupby('算法名称')['执行时间'].mean().sort_values()
        
        bars = ax.bar(range(len(avg_times)), avg_times.values, 
                     color=[algorithm_style[alg][0] for alg in avg_times.index],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('算法类型', fontsize=14)
        ax.set_ylabel('平均执行时间 (ms)', fontsize=14)
        ax.set_title('算法平均执行时间比较（物品数量≤200）', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(avg_times)))
        ax.set_xticklabels(avg_times.index, rotation=0)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, avg_times.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 图5: 算法执行时间分布箱线图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备箱线图数据
        box_data = []
        box_labels = []
        for alg in algorithms:
            alg_data = df[df['算法名称'] == alg]['执行时间']
            if len(alg_data) > 0:
                box_data.append(alg_data)
                box_labels.append(alg)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # 设置箱线图颜色
            for patch, alg in zip(bp['boxes'], box_labels):
                patch.set_facecolor(algorithm_style[alg][0])
                patch.set_alpha(0.7)
        
        ax.set_xlabel('算法类型', fontsize=14)
        ax.set_ylabel('执行时间 (ms)', fontsize=14)
        ax.set_title('算法执行时间分布（物品数量≤200）', fontsize=16, fontweight='bold')
        ax.set_yscale('log')  # 对数坐标更好地显示差异
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
    print(f"图表已保存到: {output_filename}")

def generate_summary_statistics(df):
    """
    生成统计摘要
    """
    print("\n=== 实验结果统计摘要（物品数量≤200）===")
    print(f"总测试案例数: {len(df)}")
    print(f"算法类型: {', '.join(sorted(df['算法名称'].unique()))}")
    print(f"物品数量范围: {df['物品数量'].min()} - {df['物品数量'].max()}")
    print(f"背包容量: {', '.join(map(str, sorted(df['背包容量'].unique())))}")
    
    # 显示包含的物品数量
    unique_items = [int(x) for x in sorted(df['物品数量'].unique())]
    print(f"测试的物品数量: {unique_items}")
    
    # 各算法平均执行时间
    print("\n各算法平均执行时间 (ms):")
    avg_times = df.groupby('算法名称')['执行时间'].agg(['mean', 'std', 'min', 'max', 'count'])
    for alg in avg_times.index:
        stats = avg_times.loc[alg]
        print(f"  {alg}: 平均 {stats['mean']:.3f} ± {stats['std']:.3f}, 范围 [{stats['min']:.3f}, {stats['max']:.3f}], 测试次数 {stats['count']}")
    
    # 按物品数量统计
    print("\n按物品数量统计:")
    item_stats = df.groupby('物品数量')['执行时间'].agg(['mean', 'count'])
    unique_items = [int(x) for x in sorted(df['物品数量'].unique())]
    print(f"物品数量分布: {unique_items}")
    
    # 显示性能对比摘要
    print("\n性能对比摘要:")
    mean_times = avg_times['mean']
    fastest = mean_times.idxmin()
    slowest = mean_times.idxmax()
    print(f"最快算法: {fastest} ({mean_times[fastest]:.3f} ms)")
    print(f"最慢算法: {slowest} ({mean_times[slowest]:.3f} ms)")
    print(f"性能差异: {mean_times[slowest]/mean_times[fastest]:.1f}倍")

def main():
    """
    主函数
    """
    filename = '实验结果汇总.csv'
    
    print("开始分析0-1背包问题算法性能数据...")
    
    # 读取数据
    df = read_and_process_data(filename)
    if df is None:
        return
    
    # 清理和处理数据
    df_clean = clean_and_process_data(df)
    if df_clean is None:
        return
    
    # 生成统计摘要
    generate_summary_statistics(df_clean)
    
    # 创建对比图表
    output_file = '0-1背包算法执行时间对比分析（物品数量≤200）.pdf'
    create_execution_time_comparison(df_clean, output_file)
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print("\n图表包含（仅物品数量≤200的数据）:")
    print("1. 不同背包容量下的算法执行时间对比（对数坐标）")
    print("2. 算法性能随规模变化趋势分析")
    print("3. 算法执行时间热力图")
    print("4. 算法平均执行时间比较")
    print("5. 算法执行时间分布箱线图")
    print(f"\n注意: 已过滤掉物品数量>200的{252 - len(df_clean)}条记录")
    print("回溯法和蛮力法的零执行时间数据点已在图表中过滤")

if __name__ == "__main__":
    main()