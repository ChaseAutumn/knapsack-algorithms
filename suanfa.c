#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>

// Windows兼容性处理
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

// 增大最大限制以支持32万个物品
#define MAX_ITEMS 500000
#define MAX_CAPACITY 10000000

typedef struct {
    int id;
    double weight;
    double value;
    double ratio; // 价值重量比
} Item;

typedef struct {
    double value;
    int* solution;
    double execution_time; // 执行时间(毫秒)
    double memory_usage;   // 内存使用量(MB)
    int is_valid;          // 解是否有效
} Result;

typedef struct {
    double ratio;
    int index;
} RatioIndex;

// 全局变量 - 改为动态分配避免栈溢出
Item* items = NULL;
int n;
double capacity;

// 回溯法相关全局变量
double backtrack_best;
int* backtrack_result = NULL;
int* current_solution = NULL;

// 时间记录变量
long long program_start_time;
long long program_end_time;
long long total_algorithm_time = 0;

// 函数声明
void* safe_malloc(size_t size);
void* safe_calloc(size_t num, size_t size);
long long get_time_microseconds(void);
double get_memory_usage(void);
void initialize_global_arrays(int max_items);
void cleanup_global_arrays(void);
Result init_result(void);
int compare_ratio_desc(const void* a, const void* b);
Result brute_force(void);
Result dynamic_programming(void);
Result greedy(void);
void backtrack(int idx, double current_weight, double current_value, int* index_mapping);
Result backtrack_solve(void);
void generate_items(int item_count, unsigned int seed);

// 四个表格生成函数
void generate_test_data_sample_table(void);
void generate_execution_time_table(void);
void generate_solution_quality_table(void);
void generate_memory_usage_table(void);
void generate_all_tables(void);

// 其他函数声明
void print_selected_items(Result result, const char* algorithm_name);
void print_algorithm_detailed_result(Result result, const char* algorithm_name,
    int item_count, double test_capacity, FILE* detail_file);
void brute_force_detailed_process(void);
void dynamic_programming_detailed_process(void);
void greedy_detailed_process(void);
void backtrack_detailed_process(void);
void verify_simple_example(void);
void large_scale_performance_test(void);

// 比较函数，用于排序
int compare_ratio_desc(const void* a, const void* b) {
    RatioIndex* ra = (RatioIndex*)a;
    RatioIndex* rb = (RatioIndex*)b;
    if (ra->ratio > rb->ratio) return -1;
    if (ra->ratio < rb->ratio) return 1;
    return 0;
}

// 安全的内存分配函数
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        printf("错误：内存分配失败，需要 %zu 字节\n", size);
        exit(1);
    }
    return ptr;
}

void* safe_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr && num > 0 && size > 0) {
        printf("错误：内存分配失败，需要 %zu 字节\n", num * size);
        exit(1);
    }
    return ptr;
}

// 跨平台获取当前时间（微秒）
long long get_time_microseconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return (long long)(counter.QuadPart * 1000000LL / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
#endif
}

// 获取内存使用量（MB）
double get_memory_usage(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Linux返回KB，转换为MB
#endif
}

// 初始化全局数组
void initialize_global_arrays(int max_items) {
    if (items) free(items);
    if (backtrack_result) free(backtrack_result);
    if (current_solution) free(current_solution);

    items = (Item*)safe_malloc(max_items * sizeof(Item));
    backtrack_result = (int*)safe_malloc(max_items * sizeof(int));
    current_solution = (int*)safe_malloc(max_items * sizeof(int));
}

// 清理全局数组
void cleanup_global_arrays(void) {
    if (items) {
        free(items);
        items = NULL;
    }
    if (backtrack_result) {
        free(backtrack_result);
        backtrack_result = NULL;
    }
    if (current_solution) {
        free(current_solution);
        current_solution = NULL;
    }
}

// 初始化结果结构
Result init_result(void) {
    Result result;
    result.solution = NULL;
    result.value = 0;
    result.execution_time = 0;
    result.memory_usage = 0;
    result.is_valid = 0;
    return result;
}

// 蛮力法实现
Result brute_force(void) {
    Result result = init_result();

    long long start_time = get_time_microseconds();

    if (n > 125) {
        printf("警告：物品数量过多(%d)，蛮力法将跳过（建议≤125）\n", n);
        result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
        result.memory_usage = 0.0;
        return result;
    }

    // 计算理论内存需求
    double theoretical_memory = n * sizeof(int) / (1024.0 * 1024.0); // solution数组

    result.solution = (int*)safe_calloc(n, sizeof(int));

    long long max_mask = 1LL << n;
    for (long long mask = 0; mask < max_mask; mask++) {
        double total_weight = 0, total_value = 0;

        for (int i = 0; i < n; i++) {
            if (mask & (1LL << i)) {
                total_weight += items[i].weight;
                total_value += items[i].value;
            }
        }

        if (total_weight <= capacity && total_value > result.value) {
            result.value = total_value;
            for (int i = 0; i < n; i++) {
                result.solution[i] = (mask & (1LL << i)) ? 1 : 0;
            }
        }
    }

    long long end_time = get_time_microseconds();
    result.execution_time = (end_time - start_time) / 1000.0;
    result.memory_usage = theoretical_memory;
    result.is_valid = 1;

    return result;
}

// 改进的动态规划实现 - 使用空间优化
Result dynamic_programming(void) {
    Result result = init_result();

    long long start_time = get_time_microseconds();

    // 检查输入有效性
    if (n <= 0 || capacity <= 0) {
        printf("警告：输入参数无效 (n=%d, capacity=%.0f)\n", n, capacity);
        result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
        result.memory_usage = 0.0;
        return result;
    }

    // 优化容量处理，根据规模选择不同的精度
    int scale_factor = 1;
    if (capacity >= 100000) {
        scale_factor = 100;  // 大容量时降低精度
    }
    else if (capacity >= 10000) {
        scale_factor = 10;   // 中等容量时适中精度
    }
    else {
        scale_factor = 1;    // 小容量时保持精度
    }

    long long int_capacity_ll = (long long)(capacity / scale_factor);

    // 检查整数溢出
    if (int_capacity_ll > 10000000LL) { // 限制在1千万以内
        printf("警告：容量过大，动态规划将跳过 (容量=%.0f)\n", capacity);
        result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
        result.memory_usage = 0.0;
        return result;
    }

    int int_capacity = (int)int_capacity_ll;

    // 计算理论内存需求并记录
    long long dp_memory = (long long)(int_capacity + 1) * sizeof(double) * 2; // 两个DP数组
    long long weights_memory = (long long)n * sizeof(int);
    long long solution_memory = (long long)n * sizeof(int);
    long long total_memory = dp_memory + weights_memory + solution_memory;
    double theoretical_memory_mb = total_memory / (1024.0 * 1024.0);

    // 对于大规模问题，不使用choice数组，改用重新计算的方式
    int use_choice_table = (n <= 5000 && int_capacity <= 100000);

    if (!use_choice_table) {
        printf("使用空间优化版本（大规模）\n");
        if (total_memory > 1000000000LL) { // 1GB限制
            printf("警告：动态规划需要内存过大(%.1fGB)，将跳过\n", total_memory / 1000000000.0);
            result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
            result.memory_usage = theoretical_memory_mb;
            return result;
        }
    }
    else {
        // 小规模问题：使用完整版本，记录选择路径
        long long choice_memory = (long long)n * (int_capacity + 1) * sizeof(int);
        total_memory += choice_memory;
        theoretical_memory_mb = total_memory / (1024.0 * 1024.0);

        if (total_memory > 2000000000LL) { // 2GB限制
            printf("警告：动态规划需要内存过大(%.1fGB)，将跳过\n", total_memory / 1000000000.0);
            result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
            result.memory_usage = theoretical_memory_mb;
            return result;
        }
    }

    result.solution = (int*)safe_calloc(n, sizeof(int));

    // 转换权重为整数
    int* int_weights = (int*)safe_malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int_weights[i] = (int)(items[i].weight / scale_factor);
        if (int_weights[i] <= 0) int_weights[i] = 1; // 确保权重至少为1
    }

    // 分配DP表 - 使用一维滚动数组优化空间
    double* prev = (double*)safe_calloc(int_capacity + 1, sizeof(double));
    double* curr = (double*)safe_calloc(int_capacity + 1, sizeof(double));

    // 根据规模决定是否使用选择记录表
    int** choice = NULL;

    if (use_choice_table) {
        choice = (int**)safe_malloc(n * sizeof(int*));
        for (int i = 0; i < n; i++) {
            choice[i] = (int*)safe_calloc(int_capacity + 1, sizeof(int));
        }
    }

    // 填充DP表
    for (int i = 0; i < n; i++) {
        for (int w = 0; w <= int_capacity; w++) {
            // 不选择当前物品
            curr[w] = prev[w];
            if (use_choice_table) choice[i][w] = 0;

            // 选择当前物品
            if (int_weights[i] <= w) {
                double new_value = prev[w - int_weights[i]] + items[i].value;
                if (new_value > curr[w]) {
                    curr[w] = new_value;
                    if (use_choice_table) choice[i][w] = 1;
                }
            }
        }

        // 交换prev和curr
        double* temp = prev;
        prev = curr;
        curr = temp;
    }

    result.value = prev[int_capacity];

    // 根据是否有选择表来恢复解
    if (use_choice_table) {
        // 使用选择表回溯
        int w = int_capacity;
        for (int i = n - 1; i >= 0; i--) {
            if (w >= 0 && w <= int_capacity && choice[i][w] == 1) {
                result.solution[i] = 1;
                w -= int_weights[i];
                if (w < 0) break; // 防止负数索引
            }
        }

        // 释放选择表
        for (int i = 0; i < n; i++) {
            free(choice[i]);
        }
        free(choice);
    }
    else {
        // 大规模问题：重新计算解（贪心近似）
        printf("大规模问题，使用贪心方式恢复解\n");

        // 按价值重量比排序
        RatioIndex* ratios = (RatioIndex*)safe_malloc(n * sizeof(RatioIndex));
        for (int i = 0; i < n; i++) {
            ratios[i].ratio = items[i].ratio;
            ratios[i].index = i;
        }
        qsort(ratios, n, sizeof(RatioIndex), compare_ratio_desc);

        // 贪心选择物品，总价值不超过DP的最优值
        double current_weight = 0;
        double current_value = 0;

        for (int i = 0; i < n && current_value < result.value * 0.95; i++) { // 允许5%误差
            int idx = ratios[i].index;
            if (current_weight + items[idx].weight <= capacity) {
                result.solution[idx] = 1;
                current_weight += items[idx].weight;
                current_value += items[idx].value;
            }
        }

        free(ratios);
    }

    // 释放内存
    free(prev);
    free(curr);
    free(int_weights);

    long long end_time = get_time_microseconds();
    result.execution_time = (end_time - start_time) / 1000.0;
    result.memory_usage = theoretical_memory_mb;
    result.is_valid = 1;

    return result;
}

// 贪心法实现
Result greedy(void) {
    Result result = init_result();

    long long start_time = get_time_microseconds();

    if (n <= 0) {
        result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
        result.memory_usage = 0.0;
        return result;
    }

    // 计算理论内存需求
    double theoretical_memory = (n * sizeof(int) + n * sizeof(RatioIndex)) / (1024.0 * 1024.0);

    result.solution = (int*)safe_calloc(n, sizeof(int));

    RatioIndex* ratios = (RatioIndex*)safe_malloc(n * sizeof(RatioIndex));
    for (int i = 0; i < n; i++) {
        ratios[i].ratio = items[i].ratio;
        ratios[i].index = i;
    }

    qsort(ratios, n, sizeof(RatioIndex), compare_ratio_desc);

    double total_weight = 0;
    for (int i = 0; i < n; i++) {
        int idx = ratios[i].index;
        if (total_weight + items[idx].weight <= capacity) {
            result.solution[idx] = 1;
            total_weight += items[idx].weight;
            result.value += items[idx].value;
        }
    }

    free(ratios);

    long long end_time = get_time_microseconds();
    result.execution_time = (end_time - start_time) / 1000.0;
    result.memory_usage = theoretical_memory;
    result.is_valid = 1;

    return result;
}

// 回溯法递归函数
void backtrack(int idx, double current_weight, double current_value, int* index_mapping) {
    if (idx == n) {
        if (current_value > backtrack_best) {
            backtrack_best = current_value;
            for (int i = 0; i < n; i++) {
                backtrack_result[index_mapping[i]] = current_solution[i];
            }
        }
        return;
    }

    if (current_weight > capacity) return;

    // 计算上界进行剪枝
    double upper_bound = current_value;
    double remaining_capacity = capacity - current_weight;
    for (int i = idx; i < n && remaining_capacity > 0; i++) {
        if (items[i].weight <= remaining_capacity) {
            upper_bound += items[i].value;
            remaining_capacity -= items[i].weight;
        }
        else {
            upper_bound += items[i].ratio * remaining_capacity;
            break;
        }
    }

    if (upper_bound <= backtrack_best) return;

    // 选择当前物品
    if (current_weight + items[idx].weight <= capacity) {
        current_solution[idx] = 1;
        backtrack(idx + 1, current_weight + items[idx].weight,
            current_value + items[idx].value, index_mapping);
    }

    // 不选择当前物品
    current_solution[idx] = 0;
    backtrack(idx + 1, current_weight, current_value, index_mapping);
}

// 回溯法主函数
Result backtrack_solve(void) {
    Result result = init_result();

    long long start_time = get_time_microseconds();

    if (n > 150) {
        printf("警告：物品数量过多(%d)，回溯法将跳过（建议≤150）\n", n);
        result.execution_time = (get_time_microseconds() - start_time) / 1000.0;
        result.memory_usage = 0.0;
        return result;
    }

    // 计算理论内存需求
    double theoretical_memory = (n * sizeof(int) * 4 + n * sizeof(Item) * 2) / (1024.0 * 1024.0);

    result.solution = (int*)safe_calloc(n, sizeof(int));

    backtrack_best = 0;
    memset(backtrack_result, 0, n * sizeof(int));
    memset(current_solution, 0, n * sizeof(int));

    // 按价值重量比排序
    int* indices = (int*)safe_malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // 简单冒泡排序按比值降序
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (items[indices[j]].ratio < items[indices[j + 1]].ratio) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // 创建排序后的物品数组
    Item* sorted_items = (Item*)safe_malloc(n * sizeof(Item));
    for (int i = 0; i < n; i++) {
        sorted_items[i] = items[indices[i]];
    }

    // 备份原始物品数组
    Item* original_items = (Item*)safe_malloc(n * sizeof(Item));
    memcpy(original_items, items, n * sizeof(Item));

    // 使用排序后的数组
    memcpy(items, sorted_items, n * sizeof(Item));

    backtrack(0, 0, 0, indices);

    result.value = backtrack_best;
    memcpy(result.solution, backtrack_result, n * sizeof(int));

    // 恢复原始数组
    memcpy(items, original_items, n * sizeof(Item));

    free(indices);
    free(sorted_items);
    free(original_items);

    long long end_time = get_time_microseconds();
    result.execution_time = (end_time - start_time) / 1000.0;
    result.memory_usage = theoretical_memory;
    result.is_valid = 1;

    return result;
}

// 生成测试数据
void generate_items(int item_count, unsigned int seed) {
    n = item_count;
    srand(seed);

    for (int i = 0; i < n; i++) {
        items[i].id = i + 1;
        items[i].weight = ((double)rand() / RAND_MAX) * 99.0 + 1.0; // 1-100
        items[i].value = ((double)rand() / RAND_MAX) * 900.0 + 100.0; // 100-1000

        // 保留两位小数
        items[i].weight = round(items[i].weight * 100) / 100.0;
        items[i].value = round(items[i].value * 100) / 100.0;
        items[i].ratio = items[i].value / items[i].weight;
    }
}

// 生成测试数据示例表格
void generate_test_data_sample_table(void) {
    printf("\n=== 3.1 测试数据示例 ===\n");
    printf("以下为容量10000背包的1000个物品统计信息示例：\n\n");

    // 生成1000个物品的数据
    generate_items(1000, 42);
    capacity = 10000.0;

    printf("| 物品编号 | 物品重量 | 物品价值 |\n");
    printf("|----------|----------|----------|\n");

    // 显示前5个物品
    for (int i = 0; i < 5; i++) {
        printf("| %-8d | %-8.2f | %-8.2f |\n",
            items[i].id, items[i].weight, items[i].value);
    }

    printf("| ...      | ...      | ...      |\n");

    // 显示后5个物品
    for (int i = 995; i < 1000; i++) {
        printf("| %-8d | %-8.2f | %-8.2f |\n",
            items[i].id, items[i].weight, items[i].value);
    }

    // 计算统计信息
    double sum_weight = 0, sum_value = 0, sum_ratio = 0;
    double sum_weight_sq = 0, sum_value_sq = 0, sum_ratio_sq = 0;

    for (int i = 0; i < n; i++) {
        sum_weight += items[i].weight;
        sum_value += items[i].value;
        sum_ratio += items[i].ratio;
        sum_weight_sq += items[i].weight * items[i].weight;
        sum_value_sq += items[i].value * items[i].value;
        sum_ratio_sq += items[i].ratio * items[i].ratio;
    }

    double avg_weight = sum_weight / n;
    double avg_value = sum_value / n;
    double avg_ratio = sum_ratio / n;

    double std_weight = sqrt(sum_weight_sq / n - avg_weight * avg_weight);
    double std_value = sqrt(sum_value_sq / n - avg_value * avg_value);
    double std_ratio = sqrt(sum_ratio_sq / n - avg_ratio * avg_ratio);

    printf("\n**数据特征统计：**\n");
    printf("· 平均重量: %.2f ± %.2f\n", avg_weight, std_weight);
    printf("· 平均价值: %.2f ± %.2f\n", avg_value, std_value);
    printf("· 平均价值重量比: %.2f ± %.2f\n", avg_ratio, std_ratio);
    printf("===============================================\n");
}

// 生成算法执行时间统计表格
void generate_execution_time_table(void) {
    printf("\n=== 3.2.1 执行时间统计（单位：毫秒）===\n");

    // 定义测试规模
    int test_sizes[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 500, 1000 };
    double test_capacity = 10000.0;
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("| 物品数量 | 容量  | 蛮力法    | 动态规划法 | 贪心法 | 回溯法  |\n");
    printf("|----------|-------|-----------|------------|--------|----------|\n");

    for (int i = 0; i < num_tests; i++) {
        int item_count = test_sizes[i];
        capacity = test_capacity;

        // 生成测试数据
        generate_items(item_count, 42);

        // 测试各算法
        Result brute_result = (item_count <= 25) ? brute_force() : init_result();
        Result dp_result = dynamic_programming();
        Result greedy_result = greedy();
        Result backtrack_result = (item_count <= 100) ? backtrack_solve() : init_result();

        printf("| %-8d | %-5.0f | ", item_count, test_capacity);

        if (brute_result.is_valid) {
            printf("%-9.2f | ", brute_result.execution_time);
        }
        else {
            printf("%-9s | ", "-");
        }

        if (dp_result.is_valid) {
            printf("%-10.2f | ", dp_result.execution_time);
        }
        else {
            printf("%-10s | ", "-");
        }

        if (greedy_result.is_valid) {
            printf("%-6.2f | ", greedy_result.execution_time);
        }
        else {
            printf("%-6s | ", "-");
        }

        if (backtrack_result.is_valid) {
            printf("%-8.2f |\n", backtrack_result.execution_time);
        }
        else {
            printf("%-8s |\n", "-");
        }

        // 释放内存
        if (brute_result.solution) free(brute_result.solution);
        if (dp_result.solution) free(dp_result.solution);
        if (greedy_result.solution) free(greedy_result.solution);
        if (backtrack_result.solution) free(backtrack_result.solution);
    }

    printf("\n**说明：** \"-\" 表示算法在合理时间内无法完成或内存不足\n");
    printf("===============================================\n");
}

// 生成解质量对比表格
void generate_solution_quality_table(void) {
    printf("\n=== 3.2.2 解质量对比 ===\n");

    // 定义测试规模
    int test_sizes[] = { 100, 500, 1000, 2000, 5000 };
    double test_capacity = 10000.0;
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("| 物品数量 | 动态规划解值 | 贪心法解值 | 解质量比 | 贪心准确率 |\n");
    printf("|----------|--------------|------------|----------|------------|\n");

    for (int i = 0; i < num_tests; i++) {
        int item_count = test_sizes[i];
        capacity = test_capacity;

        // 生成测试数据
        generate_items(item_count, 42);

        // 测试动态规划和贪心算法
        Result dp_result = dynamic_programming();
        Result greedy_result = greedy();

        if (dp_result.is_valid && greedy_result.is_valid && dp_result.value > 0) {
            double quality_ratio = greedy_result.value / dp_result.value;
            double accuracy = quality_ratio * 100.0;

            printf("| %-8d | %-12.2f | %-10.2f | %-8.4f | %-9.2f%% |\n",
                item_count, dp_result.value, greedy_result.value,
                quality_ratio, accuracy);
        }
        else {
            printf("| %-8d | %-12s | %-10s | %-8s | %-10s |\n",
                item_count, "Error", "Error", "-", "-");
        }

        // 释放内存
        if (dp_result.solution) free(dp_result.solution);
        if (greedy_result.solution) free(greedy_result.solution);
    }

    printf("\n**说明：** 解质量比 = 贪心解值/动态规划解值，贪心准确率表示接近最优解的程度\n");
    printf("===============================================\n");
}

// 生成内存使用分析表格
void generate_memory_usage_table(void) {
    printf("\n=== 3.2.3 内存使用分析（单位：MB）===\n");

    // 定义测试规模
    int test_sizes[] = { 10, 15, 20, 25, 100, 500, 1000 };
    double test_capacity = 10000.0;
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("| 物品数量 | 容量  | 蛮力法 | 动态规划法 | 贪心法 | 回溯法 |\n");
    printf("|----------|-------|--------|------------|--------|--------|\n");

    for (int i = 0; i < num_tests; i++) {
        int item_count = test_sizes[i];
        capacity = test_capacity;

        // 生成测试数据
        generate_items(item_count, 42);

        // 测试各算法
        Result brute_result = (item_count <= 25) ? brute_force() : init_result();
        Result dp_result = dynamic_programming();
        Result greedy_result = greedy();
        Result backtrack_result = (item_count <= 100) ? backtrack_solve() : init_result();

        printf("| %-8d | %-5.0f | ", item_count, test_capacity);

        if (brute_result.is_valid) {
            printf("%-6.2f | ", brute_result.memory_usage);
        }
        else {
            printf("%-6s | ", "-");
        }

        if (dp_result.is_valid) {
            printf("%-10.2f | ", dp_result.memory_usage);
        }
        else {
            printf("%-10s | ", "-");
        }

        if (greedy_result.is_valid) {
            printf("%-6.2f | ", greedy_result.memory_usage);
        }
        else {
            printf("%-6s | ", "-");
        }

        if (backtrack_result.is_valid) {
            printf("%-6.2f |\n", backtrack_result.memory_usage);
        }
        else {
            printf("%-6s |\n", "-");
        }

        // 释放内存
        if (brute_result.solution) free(brute_result.solution);
        if (dp_result.solution) free(dp_result.solution);
        if (greedy_result.solution) free(greedy_result.solution);
        if (backtrack_result.solution) free(backtrack_result.solution);
    }

    printf("\n**空间复杂度分析：**\n");
    printf("· 蛮力法：O(n) - 仅存储当前最优解\n");
    printf("· 动态规划法：O(n×W) - 存储DP表，W为背包容量（已缩放）\n");
    printf("· 贪心法：O(n) - 存储排序后的物品索引\n");
    printf("· 回溯法：O(n) - 存储递归栈和当前解\n");
    printf("\n**内存计算说明：**\n");
    printf("· 使用理论计算值，避免系统内存波动影响\n");
    printf("· 动态规划在大容量时会进行精度缩放以节省内存\n");
    printf("· 内存使用量包括算法核心数据结构的实际占用\n");
    printf("===============================================\n");
}

// 生成所有表格的综合函数
void generate_all_tables(void) {
    printf("\n*** 0-1背包问题算法性能分析报告 ***\n");
    printf("===========================================\n");

    // 生成四个表格
    generate_test_data_sample_table();
    generate_execution_time_table();
    generate_solution_quality_table();
    generate_memory_usage_table();

    // 输出到文件
    FILE* table_file = fopen("算法性能分析表格.txt", "w");
    if (table_file) {
        fprintf(table_file, "0-1背包问题算法性能分析报告\n");
        fprintf(table_file, "=======================================\n\n");
        fprintf(table_file, "该报告包含四个主要分析表格：\n");
        fprintf(table_file, "1. 测试数据示例表 - 展示实验数据格式和特征\n");
        fprintf(table_file, "2. 算法执行时间统计表 - 对比四种算法的时间效率\n");
        fprintf(table_file, "3. 解质量对比表 - 评估贪心算法的近似效果\n");
        fprintf(table_file, "4. 内存使用分析表 - 分析算法的空间复杂度\n\n");
        fprintf(table_file, "详细内容请参考控制台输出。\n");
        fclose(table_file);
        printf("表格分析报告已保存到文件：算法性能分析表格.txt\n");
    }
}

// 输出选择的物品详情（用于简单验证）
void print_selected_items(Result result, const char* algorithm_name) {
    if (!result.is_valid || !result.solution) {
        printf("%s: 未找到有效解决方案\n", algorithm_name);
        return;
    }

    double total_weight = 0, total_value = 0;
    int selected_count = 0;

    // 统计选择的物品
    for (int i = 0; i < n; i++) {
        if (result.solution[i] == 1) {
            selected_count++;
            total_weight += items[i].weight;
            total_value += items[i].value;
        }
    }

    printf("\n=== %s 求解结果 ===\n", algorithm_name);
    printf("选择物品数量: %d\n", selected_count);
    printf("总重量: %.2f / %.2f\n", total_weight, capacity);
    printf("总价值: %.2f\n", total_value);
    printf("执行时间: %.3f ms\n", result.execution_time);
    printf("内存使用: %.2f MB\n", result.memory_usage);

    if (selected_count <= 50 && selected_count > 0) { // 显示前50个物品
        printf("选择的物品详情:\n");
        printf("编号\t重量\t价值\n");
        for (int i = 0; i < n; i++) {
            if (result.solution[i] == 1) {
                printf("%d\t%.2f\t%.2f\n", items[i].id, items[i].weight, items[i].value);
            }
        }
    }
    else if (selected_count > 50) {
        printf("选择物品过多，仅显示前50个物品编号: ");
        int count = 0;
        for (int i = 0; i < n && count < 50; i++) {
            if (result.solution[i] == 1) {
                printf("%d ", items[i].id);
                count++;
            }
        }
        printf("...\n");
    }
    printf("=============================\n\n");
}

// 输出算法详细结果（第3点要求的六个内容）
void print_algorithm_detailed_result(Result result, const char* algorithm_name,
    int item_count, double test_capacity, FILE* detail_file) {
    if (!result.is_valid || !result.solution) {
        printf("\n--- %s 结果 ---\n", algorithm_name);
        printf("算法执行失败或被跳过\n");
        printf("执行时间: %.3f ms\n", result.execution_time);
        printf("占用空间: %.2f MB\n", result.memory_usage);
        printf("================\n\n");

        fprintf(detail_file, "\n--- %s 结果 ---\n", algorithm_name);
        fprintf(detail_file, "算法执行失败或被跳过\n");
        fprintf(detail_file, "执行时间: %.3f ms\n", result.execution_time);
        fprintf(detail_file, "占用空间: %.2f MB\n", result.memory_usage);
        fprintf(detail_file, "================\n\n");
        return;
    }

    // 统计选择的物品
    double total_weight = 0, total_value = 0;
    int selected_count = 0;
    int* selected_items = (int*)safe_malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        if (result.solution[i] == 1) {
            selected_items[selected_count] = items[i].id;
            selected_count++;
            total_weight += items[i].weight;
            total_value += items[i].value;
        }
    }

    // 1. 输出选择的物品编号
    printf("\n--- %s 结果 ---\n", algorithm_name);
    printf("1. 选择的物品编号: ");
    if (selected_count == 0) {
        printf("无");
    }
    else if (selected_count <= 50) {
        for (int i = 0; i < selected_count; i++) {
            printf("%d", selected_items[i]);
            if (i < selected_count - 1) printf(", ");
        }
    }
    else {
        for (int i = 0; i < 50; i++) {
            printf("%d", selected_items[i]);
            if (i < 49) printf(", ");
        }
        printf("... (共%d个物品)", selected_count);
    }
    printf("\n");

    // 2. 输出总重量
    printf("2. 总重量: %.2f / %.0f\n", total_weight, test_capacity);

    // 3. 输出总价值  
    printf("3. 物品装入背包获得的总价值: %.2f\n", total_value);

    // 4. 输出执行时间
    printf("4. 执行时间: %.3f ms\n", result.execution_time);

    // 5. 输出占用空间
    printf("5. 占用空间: %.2f MB\n", result.memory_usage);

    // 6. 输出详细的物品价值信息
    printf("6. 选择物品的详细价值:\n");
    if (selected_count <= 20) {
        printf("   编号  重量   价值\n");
        printf("   ---------------\n");
        for (int i = 0; i < n; i++) {
            if (result.solution[i] == 1) {
                printf("   %-4d  %-5.2f  %-5.2f\n",
                    items[i].id, items[i].weight, items[i].value);
            }
        }
    }
    else {
        printf("   选择物品过多，仅显示统计信息\n");
        printf("   选择物品数量: %d\n", selected_count);
        if (selected_count > 0) {
            printf("   平均重量: %.2f\n", total_weight / selected_count);
            printf("   平均价值: %.2f\n", total_value / selected_count);
        }
    }
    printf("================\n\n");

    // 同样的信息写入文件
    fprintf(detail_file, "\n--- %s 结果 ---\n", algorithm_name);
    fprintf(detail_file, "1. 选择的物品编号: ");
    if (selected_count == 0) {
        fprintf(detail_file, "无");
    }
    else if (selected_count <= 100) {
        for (int i = 0; i < selected_count; i++) {
            fprintf(detail_file, "%d", selected_items[i]);
            if (i < selected_count - 1) fprintf(detail_file, ", ");
        }
    }
    else {
        for (int i = 0; i < 100; i++) {
            fprintf(detail_file, "%d", selected_items[i]);
            if (i < 99) fprintf(detail_file, ", ");
        }
        fprintf(detail_file, "... (共%d个物品)", selected_count);
    }
    fprintf(detail_file, "\n");
    fprintf(detail_file, "2. 总重量: %.2f / %.0f\n", total_weight, test_capacity);
    fprintf(detail_file, "3. 物品装入背包获得的总价值: %.2f\n", total_value);
    fprintf(detail_file, "4. 执行时间: %.3f ms\n", result.execution_time);
    fprintf(detail_file, "5. 占用空间: %.2f MB\n", result.memory_usage);
    fprintf(detail_file, "6. 选择物品数量: %d\n", selected_count);
    if (selected_count <= 50) {
        fprintf(detail_file, "   详细物品信息:\n");
        fprintf(detail_file, "   编号  重量   价值\n");
        fprintf(detail_file, "   ---------------\n");
        for (int i = 0; i < n; i++) {
            if (result.solution[i] == 1) {
                fprintf(detail_file, "   %-4d  %-5.2f  %-5.2f\n",
                    items[i].id, items[i].weight, items[i].value);
            }
        }
    }
    fprintf(detail_file, "================\n\n");

    free(selected_items);
}

// 蛮力法详细求解过程
void brute_force_detailed_process(void) {
    printf("\n=== 蛮力法详细求解过程 ===\n");
    printf("策略：枚举所有2^5=32种可能的组合，找出满足重量限制的最优解\n");
    printf("时间复杂度：O(2^n) = O(2^5) = O(32)\n\n");

    printf("枚举过程（显示所有可能的组合）：\n");
    printf("%-8s %-15s %-8s %-8s %-10s %-10s\n", "编号", "选择组合", "重量", "价值", "是否可行", "备注");
    printf("--------------------------------------------------------------------\n");

    double best_value = 0;
    int best_combination = -1;
    long long max_mask = 1LL << n;

    for (long long mask = 0; mask < max_mask; mask++) {
        double total_weight = 0, total_value = 0;

        // 计算当前组合的重量和价值
        for (int i = 0; i < n; i++) {
            if (mask & (1LL << i)) {
                total_weight += items[i].weight;
                total_value += items[i].value;
            }
        }

        // 输出组合信息
        char combination[20] = "";
        sprintf(combination, "[");
        for (int i = 0; i < n; i++) {
            if (i > 0) strcat(combination, ",");
            strcat(combination, (mask & (1LL << i)) ? "1" : "0");
        }
        strcat(combination, "]");

        const char* feasible = (total_weight <= capacity) ? "可行" : "不可行";
        const char* note = "";

        if (total_weight <= capacity && total_value > best_value) {
            best_value = total_value;
            best_combination = (int)mask;
            note = "当前最优";
        }
        else if (total_weight > capacity) {
            note = "超重";
        }

        printf("%-8lld %-15s %-8.1f %-8.1f %-10s %-10s\n",
            mask, combination, total_weight, total_value, feasible, note);
    }

    printf("\n最终结果：\n");
    printf("最优组合编号：%d\n", best_combination);
    printf("最优价值：%.1f\n", best_value);
    printf("选择的物品：");
    for (int i = 0; i < n; i++) {
        if (best_combination & (1LL << i)) {
            printf("%d ", items[i].id);
        }
    }
    printf("\n============================\n");
}

// 动态规划详细求解过程
void dynamic_programming_detailed_process(void) {
    printf("\n=== 动态规划法详细求解过程 ===\n");
    printf("策略：构建DP表，dp[i][w]表示考虑前i个物品在容量w下的最大价值\n");
    printf("时间复杂度：O(n×W) = O(5×10) = O(50)\n");
    printf("状态转移方程：dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]]+value[i])\n\n");

    // 创建DP表用于显示
    int W = (int)capacity;
    if (W > 10) W = 10; // 限制显示大小

    double dp[6][11]; // 0-5个物品, 0-10容量
    int choice[6][11]; // 记录选择

    // 初始化
    for (int i = 0; i <= n && i <= 5; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = 0;
            choice[i][w] = 0;
        }
    }

    printf("DP表构建过程：\n");
    printf("容量→  ");
    for (int w = 0; w <= W; w++) {
        printf("%3d ", w);
    }
    printf("\n");

    // 填充DP表
    for (int i = 1; i <= n && i <= 5; i++) {
        printf("物品%d  ", i);
        for (int w = 0; w <= W; w++) {
            // 不选择当前物品
            dp[i][w] = dp[i - 1][w];
            choice[i][w] = 0;

            // 选择当前物品
            if (items[i - 1].weight <= w) {
                double new_value = dp[i - 1][(int)(w - items[i - 1].weight)] + items[i - 1].value;
                if (new_value > dp[i][w]) {
                    dp[i][w] = new_value;
                    choice[i][w] = 1;
                }
            }
            printf("%3.0f ", dp[i][w]);
        }
        printf("\n");
    }

    printf("\n最终结果：最大价值=%.0f\n", dp[n <= 5 ? n : 5][W]);
    printf("===============================\n");
}

// 贪心法详细求解过程
void greedy_detailed_process(void) {
    printf("\n=== 贪心法详细求解过程 ===\n");
    printf("策略：按价值重量比从大到小排序，依次选择能装入背包的物品\n");
    printf("时间复杂度：O(n log n) = O(5 log 5) ≈ O(12)\n\n");

    printf("排序前的物品信息：\n");
    printf("编号  重量  价值  价值/重量比\n");
    printf("---------------------------\n");
    for (int i = 0; i < n && i < 5; i++) {
        printf("%-4d  %-4.1f  %-4.1f  %-8.2f\n",
            items[i].id, items[i].weight, items[i].value, items[i].ratio);
    }

    printf("\n最终结果显示在上方的表格中\n");
    printf("===========================\n");
}

// 回溯法详细求解过程
void backtrack_detailed_process(void) {
    printf("\n=== 回溯法详细求解过程 ===\n");
    printf("策略：深度优先搜索 + 分支限界剪枝\n");
    printf("时间复杂度：O(2^n)，但通过剪枝可显著减少搜索空间\n\n");

    printf("搜索策略说明：\n");
    printf("1. 分支限界：计算当前节点的价值上界\n");
    printf("2. 上界计算：贪心方式填满剩余容量的理论最大价值\n");
    printf("3. 剪枝条件：上界≤当前已知最优解 或 重量>背包容量\n");
    printf("4. 搜索顺序：按价值重量比降序排列，优先搜索高价值比物品\n");

    printf("\n搜索空间优化效果：\n");
    printf("理论搜索空间：2^n 个节点\n");
    printf("实际搜索节点：通过剪枝大幅减少\n");
    printf("============================\n");
}

// 验证简单示例的正确性
void verify_simple_example(void) {
    printf("=== 0-1背包问题算法正确性验证 ===\n");
    printf("问题描述：背包容量为10，物品重量为[2, 2, 6, 5, 4]，价值为[6, 3, 5, 4, 6]\n");
    printf("期望结果：选择物品1、2和5，获得最大价值15\n\n");

    // 创建示例数据
    n = 5;
    capacity = 10.0;

    items[0] = (Item){ 1, 2.0, 6.0, 3.0 };
    items[1] = (Item){ 2, 2.0, 3.0, 1.5 };
    items[2] = (Item){ 3, 6.0, 5.0, 0.83 };
    items[3] = (Item){ 4, 5.0, 4.0, 0.8 };
    items[4] = (Item){ 5, 4.0, 6.0, 1.5 };

    printf("物品详情：\n");
    printf("编号\t重量\t价值\t价值/重量比\n");
    printf("-------------------------------\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%.1f\t%.1f\t%.2f\n",
            items[i].id, items[i].weight, items[i].value, items[i].ratio);
    }
    printf("\n");

    // 各算法详细求解过程
    brute_force_detailed_process();
    dynamic_programming_detailed_process();
    greedy_detailed_process();
    backtrack_detailed_process();

    printf("\n=== 算法结果对比 ===\n");

    // 记录算法执行时间开始
    long long algo_start = get_time_microseconds();

    // 测试各算法
    Result brute_result = brute_force();
    Result dp_result = dynamic_programming();
    Result greedy_result = greedy();
    Result backtrack_result = backtrack_solve();

    // 记录算法执行时间结束
    long long algo_end = get_time_microseconds();
    total_algorithm_time += (algo_end - algo_start);

    print_selected_items(brute_result, "蛮力法");
    print_selected_items(dp_result, "动态规划法");
    print_selected_items(greedy_result, "贪心法");
    print_selected_items(backtrack_result, "回溯法");

    // 算法性能对比表
    printf("=== 算法性能对比表 ===\n");
    printf("算法名称     时间复杂度      空间复杂度    最优性   本例执行时间\n");
    printf("================================================================\n");
    if (brute_result.is_valid) {
        printf("蛮力法       O(2^n)         O(n)         是       %.3f ms\n", brute_result.execution_time);
    }
    else {
        printf("蛮力法       O(2^n)         O(n)         是       未执行\n");
    }
    if (dp_result.is_valid) {
        printf("动态规划     O(n×W)         O(n×W)       是       %.3f ms\n", dp_result.execution_time);
    }
    else {
        printf("动态规划     O(n×W)         O(n×W)       是       未执行\n");
    }
    if (greedy_result.is_valid) {
        printf("贪心法       O(n log n)     O(n)         否       %.3f ms\n", greedy_result.execution_time);
    }
    else {
        printf("贪心法       O(n log n)     O(n)         否       未执行\n");
    }
    if (backtrack_result.is_valid) {
        printf("回溯法       O(2^n)*        O(n)         是       %.3f ms\n", backtrack_result.execution_time);
    }
    else {
        printf("回溯法       O(2^n)*        O(n)         是       未执行\n");
    }
    printf("================================================================\n");
    printf("注：* 表示有剪枝优化，实际性能优于理论最坏情况\n\n");

    // 验证结果
    printf("=== 算法验证结果 ===\n");
    if (brute_result.is_valid && dp_result.is_valid && backtrack_result.is_valid &&
        fabs(brute_result.value - 15.0) < 1e-6 && fabs(dp_result.value - 15.0) < 1e-6 &&
        fabs(backtrack_result.value - 15.0) < 1e-6) {
        printf("最优算法验证通过！蛮力法、动态规划法、回溯法都找到了最优解（价值=15）\n");
    }
    else {
        printf("算法验证结果：\n");
        printf("  蛮力法: 有效=%s, 价值=%.2f\n", brute_result.is_valid ? "是" : "否", brute_result.value);
        printf("  动态规划: 有效=%s, 价值=%.2f\n", dp_result.is_valid ? "是" : "否", dp_result.value);
        printf("  回溯法: 有效=%s, 价值=%.2f\n", backtrack_result.is_valid ? "是" : "否", backtrack_result.value);
    }

    if (dp_result.is_valid && greedy_result.is_valid && dp_result.value > 0) {
        double greedy_accuracy = (greedy_result.value / dp_result.value) * 100;
        printf("贪心法准确率：%.2f%%\n", greedy_accuracy);
        if (greedy_accuracy < 100.0) {
            printf("贪心法局限性体现：未找到最优解，说明局部最优≠全局最优\n");
        }
    }

    printf("\n算法选择建议：\n");
    printf("- 小规模问题(n≤20)：推荐动态规划法，时空效率平衡\n");
    printf("- 中等规模问题(20<n≤10000)：推荐动态规划法\n");
    printf("- 大规模问题(n>10000)：推荐贪心法，速度快且解质量较好\n");
    printf("- 理论验证：蛮力法保证最优但仅限极小规模\n");
    printf("- 特殊需求：回溯法可提供搜索过程洞察\n");
    printf("=======================\n\n");

    // 释放内存
    if (brute_result.solution) free(brute_result.solution);
    if (dp_result.solution) free(dp_result.solution);
    if (greedy_result.solution) free(greedy_result.solution);
    if (backtrack_result.solution) free(backtrack_result.solution);
}

// 大规模性能测试（第2点要求）
void large_scale_performance_test(void) {
    printf("=== 大规模0-1背包问题性能测试 ===\n");

    // 第2点要求的测试参数 + 新增的小规模测试
    int item_counts[] = { 10, 20, 30, 40, 50,60,70,80,90,  100,125,150,175,200, 500, 1000, 2000, 3000, 4000, 5000 };
    double capacities[] = { 10000, 100000, 1000000 };
    int num_item_counts = sizeof(item_counts) / sizeof(item_counts[0]);
    int num_capacities = sizeof(capacities) / sizeof(capacities[0]);

    printf("测试规模: 物品数量10-5000个，背包容量10000-1000000\n");
    printf("数据范围: 重量1-100，价值100-1000（保留两位小数）\n");
    printf("算法限制: 蛮力法≤25个物品，回溯法≤100个物品，动态规划和贪心法无限制\n");
    printf("每次测试都会输出六个内容：物品编号、重量、价值、总价值、执行时间、占用空间\n\n");

    // 创建详细结果文件和汇总CSV文件
    FILE* detail_file = fopen("详细实验结果.txt", "w");
    FILE* summary_file = fopen("实验结果汇总.csv", "w");

    if (!detail_file || !summary_file) {
        printf("无法创建结果文件\n");
        return;
    }

    fprintf(detail_file, "0-1背包问题大规模性能测试详细结果\n");
    fprintf(detail_file, "=====================================\n");
    fprintf(detail_file, "算法限制说明：\n");
    fprintf(detail_file, "- 蛮力法：适用于≤25个物品（时间复杂度O(2^n)）\n");
    fprintf(detail_file, "- 回溯法：适用于≤100个物品（带剪枝优化）\n");
    fprintf(detail_file, "- 动态规划：适用于大部分规模（空间优化版本）\n");
    fprintf(detail_file, "- 贪心法：适用于所有规模（时间复杂度O(n log n)）\n\n");

    // CSV文件头部包含所有结果
    fprintf(summary_file, "物品数量,背包容量,算法名称,");
    fprintf(summary_file, "选择物品数量,总重量,总价值,执行时间(ms),占用空间(MB),");
    fprintf(summary_file, "选择的物品编号(前50个),是否找到解\n");

    // 开始大规模测试
    for (int i = 0; i < num_item_counts; i++) {
        for (int j = 0; j < num_capacities; j++) {
            int item_count = item_counts[i];
            capacity = capacities[j];

            printf("===========================================\n");
            printf("测试: %d个物品, 背包容量=%.0f\n", item_count, capacity);
            printf("===========================================\n");

            fprintf(detail_file, "===========================================\n");
            fprintf(detail_file, "测试: %d个物品, 背包容量=%.0f\n", item_count, capacity);
            fprintf(detail_file, "===========================================\n");

            // 生成测试数据（数据生成不计入算法执行时间）
            generate_items(item_count, 42);

            Result brute_result = init_result();
            Result dp_result = init_result();
            Result greedy_result = init_result();
            Result backtrack_result = init_result();

            // 记录当前测试轮次的算法执行时间开始
            long long round_algo_start = get_time_microseconds();

            // 蛮力法（适用于≤25个物品）
            if (item_count <= 25) {
                printf("正在执行蛮力法...\n");
                brute_result = brute_force();
                print_algorithm_detailed_result(brute_result, "蛮力法", item_count, capacity, detail_file);

                // 写入CSV
                if (brute_result.is_valid && brute_result.solution) {
                    int selected_count = 0;
                    double total_weight = 0, total_value = 0;
                    char selected_items_str[1000] = "";

                    for (int k = 0; k < n; k++) {
                        if (brute_result.solution[k] == 1) {
                            selected_count++;
                            total_weight += items[k].weight;
                            total_value += items[k].value;

                            if (selected_count <= 50) {
                                char item_str[20];
                                sprintf(item_str, "%d", items[k].id);
                                if (strlen(selected_items_str) + strlen(item_str) + 2 < sizeof(selected_items_str)) {
                                    if (strlen(selected_items_str) > 0) strcat(selected_items_str, ";");
                                    strcat(selected_items_str, item_str);
                                }
                            }
                        }
                    }

                    fprintf(summary_file, "%d,%.0f,蛮力法,%d,%.2f,%.2f,%.3f,%.2f,%s,是\n",
                        item_count, capacity, selected_count, total_weight, total_value,
                        brute_result.execution_time, brute_result.memory_usage, selected_items_str);
                }
                else {
                    fprintf(summary_file, "%d,%.0f,蛮力法,0,0,0,%.3f,%.2f,,否\n",
                        item_count, capacity, brute_result.execution_time, brute_result.memory_usage);
                }
            }
            else {
                printf("蛮力法: 跳过（物品数量过多，限制≤25）\n");
                fprintf(detail_file, "蛮力法: 跳过（物品数量过多，限制≤25）\n\n");
                fprintf(summary_file, "%d,%.0f,蛮力法,0,0,0,0,0,,跳过\n", item_count, capacity);
            }

            // 动态规划法（适用于大部分情况）
            printf("正在执行动态规划法...\n");
            dp_result = dynamic_programming();
            print_algorithm_detailed_result(dp_result, "动态规划法", item_count, capacity, detail_file);

            // 写入CSV - 动态规划
            if (dp_result.is_valid && dp_result.solution) {
                int selected_count = 0;
                double total_weight = 0, total_value = 0;
                char selected_items_str[1000] = "";

                for (int k = 0; k < n; k++) {
                    if (dp_result.solution[k] == 1) {
                        selected_count++;
                        total_weight += items[k].weight;
                        total_value += items[k].value;

                        if (selected_count <= 50) {
                            char item_str[20];
                            sprintf(item_str, "%d", items[k].id);
                            if (strlen(selected_items_str) + strlen(item_str) + 2 < sizeof(selected_items_str)) {
                                if (strlen(selected_items_str) > 0) strcat(selected_items_str, ";");
                                strcat(selected_items_str, item_str);
                            }
                        }
                    }
                }

                fprintf(summary_file, "%d,%.0f,动态规划法,%d,%.2f,%.2f,%.3f,%.2f,%s,是\n",
                    item_count, capacity, selected_count, total_weight, total_value,
                    dp_result.execution_time, dp_result.memory_usage, selected_items_str);
            }
            else {
                fprintf(summary_file, "%d,%.0f,动态规划法,0,0,0,%.3f,%.2f,,否\n",
                    item_count, capacity, dp_result.execution_time, dp_result.memory_usage);
            }

            // 贪心法（始终执行）
            printf("正在执行贪心法...\n");
            greedy_result = greedy();
            print_algorithm_detailed_result(greedy_result, "贪心法", item_count, capacity, detail_file);

            // 写入CSV - 贪心法
            if (greedy_result.is_valid && greedy_result.solution) {
                int selected_count = 0;
                double total_weight = 0, total_value = 0;
                char selected_items_str[1000] = "";

                for (int k = 0; k < n; k++) {
                    if (greedy_result.solution[k] == 1) {
                        selected_count++;
                        total_weight += items[k].weight;
                        total_value += items[k].value;

                        if (selected_count <= 50) {
                            char item_str[20];
                            sprintf(item_str, "%d", items[k].id);
                            if (strlen(selected_items_str) + strlen(item_str) + 2 < sizeof(selected_items_str)) {
                                if (strlen(selected_items_str) > 0) strcat(selected_items_str, ";");
                                strcat(selected_items_str, item_str);
                            }
                        }
                    }
                }

                fprintf(summary_file, "%d,%.0f,贪心法,%d,%.2f,%.2f,%.3f,%.2f,%s,是\n",
                    item_count, capacity, selected_count, total_weight, total_value,
                    greedy_result.execution_time, greedy_result.memory_usage, selected_items_str);
            }
            else {
                fprintf(summary_file, "%d,%.0f,贪心法,0,0,0,%.3f,%.2f,,否\n",
                    item_count, capacity, greedy_result.execution_time, greedy_result.memory_usage);
            }

            // 回溯法（适用于≤100个物品）
            if (item_count <= 100) {
                printf("正在执行回溯法...\n");
                backtrack_result = backtrack_solve();
                print_algorithm_detailed_result(backtrack_result, "回溯法", item_count, capacity, detail_file);

                // 写入CSV - 回溯法
                if (backtrack_result.is_valid && backtrack_result.solution) {
                    int selected_count = 0;
                    double total_weight = 0, total_value = 0;
                    char selected_items_str[1000] = "";

                    for (int k = 0; k < n; k++) {
                        if (backtrack_result.solution[k] == 1) {
                            selected_count++;
                            total_weight += items[k].weight;
                            total_value += items[k].value;

                            if (selected_count <= 50) {
                                char item_str[20];
                                sprintf(item_str, "%d", items[k].id);
                                if (strlen(selected_items_str) + strlen(item_str) + 2 < sizeof(selected_items_str)) {
                                    if (strlen(selected_items_str) > 0) strcat(selected_items_str, ";");
                                    strcat(selected_items_str, item_str);
                                }
                            }
                        }
                    }

                    fprintf(summary_file, "%d,%.0f,回溯法,%d,%.2f,%.2f,%.3f,%.2f,%s,是\n",
                        item_count, capacity, selected_count, total_weight, total_value,
                        backtrack_result.execution_time, backtrack_result.memory_usage, selected_items_str);
                }
                else {
                    fprintf(summary_file, "%d,%.0f,回溯法,0,0,0,%.3f,%.2f,,否\n",
                        item_count, capacity, backtrack_result.execution_time, backtrack_result.memory_usage);
                }
            }
            else {
                printf("回溯法: 跳过（物品数量过多，限制≤100）\n");
                fprintf(detail_file, "回溯法: 跳过（物品数量过多，限制≤100）\n\n");
                fprintf(summary_file, "%d,%.0f,回溯法,0,0,0,0,0,,跳过\n", item_count, capacity);
            }

            // 记录当前测试轮次的算法执行时间结束
            long long round_algo_end = get_time_microseconds();
            total_algorithm_time += (round_algo_end - round_algo_start);

            printf("本轮测试完成\n\n");
            fprintf(detail_file, "本轮测试完成\n\n");

            // 释放内存
            if (brute_result.solution) free(brute_result.solution);
            if (dp_result.solution) free(dp_result.solution);
            if (greedy_result.solution) free(greedy_result.solution);
            if (backtrack_result.solution) free(backtrack_result.solution);
        }
    }

    fclose(detail_file);
    fclose(summary_file);
    printf("大规模测试完成！\n");
    printf("详细结果保存在: 详细实验结果.txt\n");
    printf("汇总结果保存在: 实验结果汇总.csv (包含所有计算结果)\n\n");
}

int main(void) {
    // 第3点要求：程序开始时记录时间
    program_start_time = get_time_microseconds();

    printf("0-1背包问题算法性能测试系统 v4.1\n");
    printf("==================================\n");
    printf("支持Windows和Linux平台\n");
    printf("实现算法：蛮力法、动态规划、贪心法、回溯法\n");
    printf("算法适用范围：\n");
    printf("  - 蛮力法：≤25个物品（时间复杂度O(2^n)）\n");
    printf("  - 回溯法：≤100个物品（带剪枝优化）\n");
    printf("  - 动态规划：大部分规模（空间优化版本）\n");
    printf("  - 贪心法：所有规模（时间复杂度O(n log n)）\n");
    printf("测试功能：算法正确性验证、大规模性能测试、内存占用分析\n");
    printf("安全特性：内存溢出保护、空指针检查、异常处理\n");
    printf("输出特色：每次测试输出六个详细内容（物品编号、重量、价值、总价值、执行时间、占用空间）\n");
    printf("测试规模：10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 500, 1000, 2000, 3000, 4000, 5000 个物品\n");
    printf("新增功能：四个统计表格生成\n\n");

    // 初始化全局数组
    initialize_global_arrays(MAX_ITEMS);

    // 步骤1: 算法正确性验证
    printf("步骤1: 算法正确性验证\n");
    printf("====================\n");
    verify_simple_example();

    // 步骤2: 大规模性能测试
    printf("步骤2: 大规模性能测试\n");
    printf("==================\n");
    large_scale_performance_test();

    // 步骤3: 生成四个统计表格（新增功能）
    printf("步骤3: 生成算法性能统计表格\n");
    printf("============================\n");
    generate_all_tables();

    // 清理全局数组
    cleanup_global_arrays();

    // 第3点要求：程序结束时记录时间并计算总执行时间
    program_end_time = get_time_microseconds();
    double total_program_time = (program_end_time - program_start_time) / 1000.0; // 程序总时间
    double pure_algorithm_time = total_algorithm_time / 1000.0; // 纯算法执行时间

    printf("=== 程序执行完成 ===\n");
    printf("程序总运行时间: %.3f ms (%.3f 秒)\n", total_program_time, total_program_time / 1000.0);
    printf("纯算法执行时间: %.3f ms (%.3f 秒) [不包含数据生成时间]\n", pure_algorithm_time, pure_algorithm_time / 1000.0);
    printf("数据生成等其他时间: %.3f ms\n", total_program_time - pure_algorithm_time);
    printf("算法时间占比: %.1f%%\n", (pure_algorithm_time / total_program_time) * 100);
    printf("\n生成的文件:\n");
    printf("- 详细实验结果.txt: 详细的测试过程和六个输出内容\n");
    printf("- 实验结果汇总.csv: CSV格式的完整汇总数据\n");
    printf("- 算法性能分析表格.txt: 四个统计表格的分析报告\n");
    printf("\n测试完成！按任意键退出...\n");

    // 等待用户输入
    getchar();

    return 0;
}