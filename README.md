# knapsack-algorithms
# 0-1背包问题算法性能分析

## 项目简介

本项目对0-1背包问题的四种经典算法进行了全面的性能分析和比较研究，包括蛮力法、动态规划法、贪心法和回溯法。通过大量实验数据分析各算法在不同规模和容量下的时间复杂度、空间复杂度和解质量表现。

## 算法实现

### 🔍 算法概述
- **蛮力法（Brute Force）**：枚举所有可能组合，保证最优解
- **动态规划法（Dynamic Programming）**：基于子问题最优性的精确算法
- **贪心法（Greedy Algorithm）**：基于价值密度的快速近似算法
- **回溯法（Backtracking）**：采用分支限界的搜索算法

### ⚡ 性能特征
| 算法 | 时间复杂度 | 空间复杂度 | 解质量 | 适用规模 |
|------|------------|------------|--------|----------|
| 蛮力法 | O(2^n) | O(n) | 最优解 | n ≤ 25 |
| 动态规划 | O(n×W) | O(n×W) | 最优解 | 中等规模 |
| 贪心法 | O(n log n) | O(n) | 近似解(94%-98%) | 大规模 |
| 回溯法 | O(2^n) | O(n) | 最优解 | 中等规模 |

## 项目结构

```
knapsack-algorithms/
├── README.md                                    # 项目说明文档
├── suanfa.c                                    # 核心算法实现（C语言）
├── suanfa.sln                                  # Visual Studio解决方案文件
├── suanfa.vcxproj                              # 项目配置文件
├── suanfa.vcxproj.filters                      # 项目过滤器
├── suanfa.vcxproj.user                         # 用户配置文件
├── 实验结果汇总.csv                              # 实验数据汇总
├── 算法性能分析表格.txt                          # 性能分析数据
├── 详细实验结果.txt                              # 详细实验记录
├── 0-1背包算法执行时间对比分析(物品数量≤200).pdf   # 实验报告
├── Comparison chart of execution time.py        # 数据可视化脚本
├── .vs/                                        # Visual Studio缓存目录
└── x64/                                        # 编译输出目录


C：
.VS
suanfa
x64
 suanfa.c
suanfa.sln
suanfa.vcxproj
suanfa.vcxproj.filters
suanfa.vcxproj.user
实验结果汇总.csv
算法性能分析表格.txt
详细实验结果.txt
Python：
.vs
.vscode
0-1背包算法执行时间对比分析(物品数量≤200).pdf
Comparison chart of execution time.py实验结果汇总.csv
```

## 实验设计

### 📊 测试参数
- **物品数量范围**：10 - 5000个
- **背包容量**：10,000 - 1,000,000
- **测试轮次**：每组参数运行100次取平均值
- **评估指标**：执行时间、内存使用、解的准确率

### 🎯 测试场景
1. **小规模精确性测试**：验证算法正确性
2. **中等规模性能测试**：分析时间空间复杂度
3. **大规模可扩展性测试**：评估实际应用潜力
4. **容量敏感性测试**：研究背包容量对性能的影响

## 主要发现

### ⚡ 性能表现
- **蛮力法**：严格遵循O(2^n)复杂度，仅适用于理论验证
- **动态规划**：多项式时间复杂度，中等规模问题的最佳选择
- **贪心法**：卓越的效率表现，平均准确率96.51%
- **回溯法**：通过剪枝优化，实际性能优于理论最坏情况

### 📈 适用建议
- **小规模问题（n ≤ 100）**：推荐动态规划或回溯法
- **中等规模问题（100 < n ≤ 1000）**：推荐动态规划法
- **大规模问题（n > 1000）**：推荐贪心法
- **实时应用场景**：推荐贪心法

## 快速开始

### 环境要求
- Visual Studio 2019 或更高版本
- C编译器支持
- Python 3.x（用于数据可视化）

### 编译运行
```bash
# 1. 克隆项目
git clone https://github.com/yourusername/knapsack-algorithms.git
cd knapsack-algorithms

# 2. 使用Visual Studio打开解决方案
# 双击 suanfa.sln 文件

# 3. 编译并运行
# 在Visual Studio中按F5运行，或者
# 在开发者命令提示符中：
msbuild suanfa.sln
./x64/Debug/suanfa.exe
```

### 查看结果
```bash
# 查看实验数据汇总
cat 实验结果汇总.csv

# 查看详细实验结果
cat 详细实验结果.txt

# 生成可视化图表
python "Comparison chart of execution time.py"
```

## 实验结果

详细的实验数据和分析结果包含在以下文件中：
- `实验结果汇总.csv` - 所有测试场景的执行时间和内存使用数据
- `算法性能分析表格.txt` - 结构化的性能比较表格
- `详细实验结果.txt` - 完整的实验执行日志
- `0-1背包算法执行时间对比分析(物品数量≤200).pdf` - 综合分析报告

## 技术特色

- ✅ **多算法集成**：四种经典算法的完整实现
- ✅ **性能基准测试**：系统性的性能评估框架
- ✅ **数据可视化**：直观的性能对比图表
- ✅ **实际应用指导**：基于实验数据的算法选择建议

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**关键词**：0-1背包问题、动态规划、贪心算法、回溯法、算法性能分析、时间复杂度、空间复杂度
