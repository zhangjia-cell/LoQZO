"""
中文注释版：templates.py

这个文件的作用：
1. 定义“任务模板（prompt template）”基类 Template；
2. 为不同数据集提供各自的 prompt 构造方式；
3. 给上层训练/评估流程统一接口：
   - encode(sample): 只构造“题面/上下文”，不带答案
   - verbalize(sample, candidate): 构造“题面 + 候选答案/标准答案”
   - encode_sfc(sample): 为 SFC（calibration）构造“去掉输入内容”的校准 prompt
   - verbalize_sfc(sample, candidate): 为 SFC 构造“校准 prompt + 候选答案”

你可以把它理解成：
“同一个 Sample，针对不同任务，应该如何改写成 LLM 能读懂的自然语言提示词（prompt）”。

注意：
- 这个文件本身不负责训练，也不负责数据集加载；
- 它只负责“把样本变成字符串 prompt”；
- 真正调用这些模板的地方，通常在 tasks.py / run.py / utils.py 的 prompt 编码流程中。
"""


class Template:
    """
    所有任务模板的基类。

    约定了四个统一接口：
    1. encode(sample)
       输入一个样本，返回“不带答案”的 prompt。
       常用于推理阶段，给模型一个题面，让模型自己补全答案。

    2. verbalize(sample, candidate)
       输入样本和一个候选答案，返回“带答案”的完整 prompt。
       常用于多选/分类任务里，分别拼接每个 candidate，再比较 log-prob。

    3. encode_sfc(sample)
       为 SFC（surface form competition / calibration）构造去掉输入内容的 prompt。
       本质上是一个“去上下文”的校准模板。

    4. verbalize_sfc(sample, candidate)
       在 SFC prompt 基础上，拼接 candidate。
    """

    def encode(self, sample):
        """
        返回只包含题面/上下文的 prompt，不包含答案。
        子类必须自己实现。
        """
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """
        返回包含题面 + candidate 的完整 prompt。

        基类默认直接返回 candidate，
        但实际使用时几乎都会在子类里重写。
        """
        return candidate

    def encode_sfc(self, sample):
        """
        SFC 校准版本的 encode。

        默认返回 "<mask>"，但大多数子类都会根据任务自己重写。
        """
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """
        SFC 校准版本的 verbalize。

        默认只返回 candidate。
        """
        return candidate


class SST2Template(Template):
    """
    SST-2 情感分类模板。

    标签映射：
    0 -> terrible
    1 -> great

    也就是说，原始分类标签会被“词化（verbalize）”成自然语言词。
    """
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        # 只给出句子和引导短语，不给答案
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        # 拼上对应标签词，让模型判断这个补全是否合理
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        # SFC 时去掉原句，只保留答案插入位置
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class CopaTemplate(Template):
    """
    COPA 因果推理模板。

    COPA 有两种问题：
    - effect：问“结果是什么”
    - cause：问“原因是什么”

    所以这里会根据 question 字段决定使用：
    - " so "       表示结果
    - " because "  表示原因
    """

    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        """
        根据样本类型选择连接词。
        """
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        """
        生成基础 prompt：
        premise + conjunction

        例如：
        "The ground was wet because "
        """
        premise = sample.data["premise"].rstrip()

        # 去掉句尾句号，避免和后面的 because/so 拼起来太别扭
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]

        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction

        # 控制大小写风格
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        # 只返回题面，不加候选答案
        prompt = self.get_prompt(sample)
        return prompt 

    def capitalize(self, c):
        """
        控制 candidate 的首字母/整体大小写。

        "correct" 模式下：
        - 如果首词不是 I，则首词变小写
        - 是为了和前面的 premise + conjunction 更自然地衔接
        """
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            # 不做处理，保留原样
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        # 完整 prompt = 题面 + 处理过大小写的 candidate
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        # SFC 校准时只保留 because / so 这种连接词
        conjunction = self.get_conjucture(sample)
        return conjunction.strip() 

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class BoolQTemplate(Template):
    """
    BoolQ 问答模板，答案通常是 Yes / No。
    """

    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]

        # 若问题末尾没有问号，则补一个
        if not question.endswith("?"):
            question = question + "?"

        # 首字母大写，让 prompt 更自然
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        # 在题面后直接拼上候选答案
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"

    def encode_sfc(self, sample):
        # SFC 时不保留任何上下文
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV2(Template):
    """
    BoolQ 的另一个模板版本。

    与 V1 的区别主要是结尾加入了 "\\n\\n"。
    注意：如果源码里的双反斜杠不是复制时的转义问题，
    那这里输出的会是字面量 "\\n\\n"，而不是真正换行。
    """

    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV3(Template):
    """
    BoolQ 的第三个模板版本。

    和前面相比，这里使用了真正的换行 "\n"。
    """

    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class MultiRCTemplate(Template):
    """
    MultiRC 模板。
    这是阅读理解 + 回答是否正确的形式。

    verbalizer:
    0 -> No
    1 -> Yes
    """
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]

        # 这里 prompt 的意思是：
        # “我找到了这个答案，它对吗？”
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class CBTemplate(Template):
    """
    CB（CommitmentBank）自然语言推理模板。

    标签被 verbalize 为：
    0 -> Yes
    1 -> No
    2 -> Maybe
    """
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WICTemplate(Template):
    """
    WiC（Word-in-Context）模板。
    判断同一个词在两句话中是否同义。
    """
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n"

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WSCTemplate(Template):
    """
    WSC 共指消解模板。
    判断代词 span2 是否指代 span1。
    """
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    """
    ReCoRD 阅读理解模板。
    原任务里用 @placeholder 作为待填实体位置。
    """
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer:"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    """
    ReCoRD 的 GPT-3 风格模板。

    和上面的区别：
    - 把 @highlight 替换为列表样式 "- "
    - 把 query 里的 @placeholder 直接替换成 candidate
    """
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"{passage}\n- {query}"

        # 老版本写法保留在下面，作者注释掉了
        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"- {query}"


class RTETemplate(Template):
    """
    RTE 文本蕴含模板。
    0 -> Yes
    1 -> No
    """
    # From PromptSource 1
    verbalizer={0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SQuADv2Template(Template):
    """
    SQuAD 风格问答模板（生成式）。

    注意：
    - encode() 只构造到 "Answer:"
    - verbalize() 会把答案拼上去
    - 这里实际使用的是 sample.data['answers'][0]
      也就是“多个标准答案中只取第一个”
    """

    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        """
        虽然函数签名里有 candidate，
        但当前实现实际并没有使用传入的 candidate，
        而是直接用了 sample.data['answers'][0]。
        """
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"


    def encode_sfc(self, sample):
        # 当前作者没有为这类生成式 QA 提供 SFC 版本
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class DROPTemplate(Template):
    """
    DROP 生成式问答模板。
    和 SQuAD 类似，但字段名换成了 Passage / Question / Answer。
    """

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        """
        同样，这里虽然有 candidate 参数，
        但实际使用的是 sample.data['answers'][0]。
        """
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"


    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class WinoGrandeTemplate(Template):
    """
    WinoGrande 模板。

    原句里会有一个 "_" 作为空缺位置，例如：
        "The trophy doesn't fit in the suitcase because _ is too big."
    这里会把句子切成：
    - context: "_" 前面的部分
    - target : "_" 后面的部分
    然后 verbalize 时把 candidate 插进去。
    """

    @staticmethod
    def get_prompt(sample):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = sample.data["sentence"]
        context, target = sentence.split("_")
        return context

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + candidate

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class WikiTextTemplate(Template):
    """
    WikiText 续写模板（生成任务）。

    这里把文本拆成两部分：
    - context：前半段
    - continuation：后半段

    encode() 只提供上下文，
    verbalize() 提供完整“上下文 + 续写”。
    """

    def encode(self, sample):
        context = sample.data["context"]
        return f"Continue the following text:\n\n{context}"

    def verbalize(self, sample, candidate):
        context = sample.data["context"]
        continuation = sample.data["continuation"]
        return f"Continue the following text:\n\n{context}{continuation}"

    def encode_sfc(self, sample):
        return "Continue the following text:\n\n"

    def verbalize_sfc(self, sample, candidate):
        return f"Continue the following text:\n\n{candidate}"
