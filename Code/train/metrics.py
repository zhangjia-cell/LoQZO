import numpy as np
import collections  # 这个导入在当前文件里实际上没有被使用，可删可留
import re
import string
from collections import Counter


def normalize_answer(s):
    """
    统一答案字符串的格式，便于后续做 EM / F1 评估。

    这个函数会把不同表面形式、但语义相同的答案尽量规整到一致格式。
    例如：
        "The Apple." -> "apple"

    处理步骤：
    1) 全部转为小写
    2) 去掉标点符号
    3) 去掉英文冠词 a / an / the
    4) 去掉多余空格
    """

    def remove_articles(text):
        # 去掉英文冠词：a / an / the
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # 把多个空格压成一个空格，并去掉首尾空格
        return ' '.join(text.split())

    def remove_punc(text):
        # 去掉所有英文标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 全部转小写
        return text.lower()

    # 注意：执行顺序是 先小写 -> 去标点 -> 去冠词 -> 修正空格
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_metric(predictions, metric_name):
    """
    根据 metric_name 计算评测指标。

    参数：
        predictions: 一个 Prediction 对象列表。
                     每个元素通常至少包含：
                     - pred.correct_candidate   真实答案
                     - pred.predicted_candidate 模型预测答案
        metric_name: 指标名，可选：
                     - "accuracy"
                     - "em"
                     - "f1"

    返回：
        一个标量（所有样本上的平均指标值）
    """

    if metric_name == "accuracy":
        # 分类 / 多选任务常用指标
        # --------------------------------------------------
        # 两种情况：
        # 1) correct_candidate 是单个正确答案
        # 2) correct_candidate 是一个列表，表示多个答案都算正确
        #    例如某些数据集可能允许多个等价选项
        # --------------------------------------------------
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([
                pred.predicted_candidate in pred.correct_candidate
                for pred in predictions
            ])
        else:
            return np.mean([
                pred.correct_candidate == pred.predicted_candidate
                for pred in predictions
            ])

    elif metric_name == "em":
        # 精确匹配（Exact Match）
        # 常用于问答任务。
        # 只要预测答案在 normalize 之后，与任一标准答案完全一致，就记为 1，否则记为 0。
        return np.mean([
            any([
                normalize_answer(ans) == normalize_answer(pred.predicted_candidate)
                for ans in pred.correct_candidate
            ])
            for pred in predictions
        ])

    elif metric_name == "f1":
        # F1 指标
        # 常用于问答任务（如 SQuAD / DROP）
        # 做法是：
        # 1) 把预测答案和真实答案都 normalize
        # 2) 再按 token 切分
        # 3) 计算 token-level 的 precision / recall / F1
        # 4) 如果一个样本有多个标准答案，取其中最大的 F1
        f1 = []

        for pred in predictions:
            all_f1s = []

            # 特殊情况：无答案任务
            # 这里把 "CANNOTANSWER" 和 "no answer" 单独处理，
            # 只做严格匹配，不走 token-level F1。
            if pred.correct_candidate[0] == "CANNOTANSWER" or pred.correct_candidate[0] == "no answer":
                f1.append(
                    int(
                        normalize_answer(pred.correct_candidate[0])
                        == normalize_answer(pred.predicted_candidate)
                    )
                )
            else:
                # 对每一个可能的标准答案都算一次 F1，最后取最大值
                for ans in pred.correct_candidate:
                    prediction_tokens = normalize_answer(pred.predicted_candidate).split()
                    ground_truth_tokens = normalize_answer(ans).split()

                    # Counter & Counter 表示多重集合交集，
                    # 可以正确统计重复 token 的重叠数
                    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                    num_same = sum(common.values())

                    if num_same == 0:
                        all_f1s.append(0)
                    else:
                        precision = 1.0 * num_same / len(prediction_tokens)
                        recall = 1.0 * num_same / len(ground_truth_tokens)
                        all_f1s.append((2 * precision * recall) / (precision + recall))

                # 一个样本可能有多个标准答案，取最佳匹配的那个 F1
                f1.append(max(all_f1s))

        return np.mean(f1)


def f1(pred, gold):
    """
    单样本版 F1。

    这个函数和上面的 calculate_metric(..., "f1") 逻辑几乎一致，
    区别在于它只处理一个预测和一个 gold 列表。

    注释里写得很清楚：
    This separate F1 function is used as non-differentiable metric for SQuAD

    也就是说，它主要给 SQuAD 这类“不可导目标”场景使用，
    不是批量评测入口，而是单条样本的 F1 计算工具。

    参数：
        pred: 模型预测的字符串答案
        gold: 标准答案列表，例如 ["New York", "NYC"]

    返回：
        当前样本的最大 token-level F1
    """
    if gold[0] == "CANNOTANSWER" or gold[0] == "no answer":
        # 无答案时，退化成严格匹配
        return int(normalize_answer(gold[0]) == normalize_answer(pred))
    else:
        all_f1s = []
        for ans in gold:
            prediction_tokens = normalize_answer(pred).split()
            ground_truth_tokens = normalize_answer(ans).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                all_f1s.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                all_f1s.append((2 * precision * recall) / (precision + recall))

        # 多个标准答案时，返回最大 F1
        return np.max(all_f1s)