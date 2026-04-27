import os
import json
import yaml
import argparse
from typing import Optional, Tuple, Dict, Any, List
from openai import OpenAI

# 改进的 system prompt：强调先理解代码和参数，然后从用户角度描述具体需求
IMPROVED_OP_PROMPT = """You are an expert at understanding data processing code and YAML configurations, and translating them into concrete, user-facing data processing requirements.

CORE REQUIREMENT:
Generate a natural language requirement that is FUNCTIONALLY EQUIVALENT to the YAML configuration, but written from a user's perspective who has NO knowledge of code, operators, or parameters.

PROCESS:
1. Understand: Carefully read the code to understand what each operator ACTUALLY does, and how parameters affect behavior
2. Translate: Describe the data processing behavior from a user's perspective, using natural language
3. Verify: Ensure the requirement would produce the same results as the YAML configuration

RULES:
1. Describe Actual Behavior in detail
    - Understand what the code ACTUALLY does, not just what the name suggests
    - Describe all important parameters like thresholds and conditions in natural language, but avoid translating them directly
2. User Perspective
   - Write as if you are the user: "I have a dataset... I need to..."
   - Write as if you do not know the code: Never mention specific function names, parameter names, file paths, model names
3. Functional Equivalence
   - The requirement MUST result in the same data processing as the YAML configuration
   - All parameters must be accurately reflected using natural, relative descriptions
   - Preserve the exact sequence of operations

OUTPUT:
Write a well-structured paragraph(s) that reads like a user requirement document. Be detailed enough that another engineer could implement the same processing from your description.

EXAMPLE:
I have a dataset of text documents that I'm preparing for training a language model. I need to clean and filter this dataset to ensure high quality. First, remove any duplicate documents, treating text as the same even if there are differences in capitalization or non-character symbols. Then, filter out any text samples that contain too many repeated character sequences - specifically, remove any text where excessive repetition dominates the content. Also remove any text that contains inappropriate or flagged words, keeping only samples where such words appear very infrequently. Additionally, ensure that each line in the text meets a reasonable length requirement, and that each complete text sample is sufficiently long. Finally, remove near-duplicate documents using a similarity-based deduplication method that identifies documents with similar content, using a window-based approach that ignores punctuation and treats text case-insensitively.
"""



API_KEY = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="http://123.57.212.178:3333/v1",
)


def query_llm(user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """
    调用大模型：
    - system: system_prompt（任务要求全部在这里，默认使用 IMPROVED_OP_PROMPT）
    - user:   只提供算子源码 + YAML 参数等原始信息
    """
    if system_prompt is None:
        system_prompt = IMPROVED_OP_PROMPT
    try:
        completion = client.chat.completions.create(
            # model="qwen-max",
            # model = "gemini-3-pro-preview",
            # model = "gpt-5.1-2025-11-13",
            model = "gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during querying LLM: {e}")
        return None


def load_recipe(yaml_path: str) -> Dict[str, Any]:
    """
    加载 YAML recipe 文件。
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Recipe file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_all_ops(recipe: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    从 recipe 中提取所有算子的名字和参数。
    假设结构类似：
    process:
      - some_op_name:
          arg1: ...
          arg2: ...
      - another_op_name:
          arg1: ...
    """
    process_list = recipe.get("process", [])
    if not process_list:
        raise ValueError("Recipe.process is empty or missing")

    ops = []
    for idx, item in enumerate(process_list):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Unexpected process item format at index {idx}: {item}")

        op_name = list(item.keys())[0]
        op_args = item[op_name] or {}
        if not isinstance(op_args, dict):
            raise ValueError(
                f"Operator arguments must be a dict for {op_name}, got: {type(op_args)}"
            )

        ops.append((op_name, op_args))

    return ops


def load_op_code(op_code_path: str) -> str:
    """
    读取算子源码。
    """
    if not os.path.exists(op_code_path):
        raise FileNotFoundError(f"Operator code file not found: {op_code_path}")
    with open(op_code_path, "r", encoding="utf-8") as f:
        return f.read()


def find_op_code_file(op_name: str, base_dir: str) -> Optional[str]:
    """
    根据算子名称在基础目录中查找对应的代码文件。
    搜索路径：base_dir/ops/*/{op_name}.py
    """
    if not os.path.exists(base_dir):
        return None

    # 常见的算子类型目录
    op_categories = ["filter", "deduplicator", "mapper", "selector", "formatter"]
    
    for category in op_categories:
        op_file = os.path.join(base_dir, "ops", category, f"{op_name}.py")
        if os.path.exists(op_file):
            return op_file
    
    # 如果没找到，尝试在整个 ops 目录下递归搜索
    ops_dir = os.path.join(base_dir, "ops")
    if os.path.exists(ops_dir):
        for root, dirs, files in os.walk(ops_dir):
            if f"{op_name}.py" in files:
                return os.path.join(root, f"{op_name}.py")
    
    return None


def build_user_prompt(
    ops_info: List[Tuple[str, Dict[str, Any], str]],  # (op_name, op_args, op_code)
    recipe: Dict[str, Any],
) -> str:
    """
    构造发送给 LLM 的 user prompt，支持多个算子。
    强调先理解代码和参数，然后从用户角度描述具体需求。
    """
    dataset_path = recipe.get("dataset_path", "")
    export_path = recipe.get("export_path", "")
    project_name = recipe.get("project_name", "")

    # 构建所有算子的 YAML 调用片段
    process_yaml_lines = ["process:"]
    for op_name, op_args, _ in ops_info:
        op_args_yaml = yaml.dump(op_args, allow_unicode=True, sort_keys=False).rstrip()
        if op_args_yaml:
            op_args_yaml_indented = op_args_yaml.replace("\n", "\n      ")
            process_yaml_lines.append(f"  - {op_name}:")
            process_yaml_lines.append(f"      {op_args_yaml_indented}")
        else:
            process_yaml_lines.append(f"  - {op_name}:")

    process_yaml = "\n".join(process_yaml_lines)

    # 构建所有算子的代码部分
    op_codes_section = []
    for idx, (op_name, _, op_code) in enumerate(ops_info, 1):
        op_codes_section.append(f"[Operator {idx}: {op_name} Python source code]")
        op_codes_section.append("```python")
        op_codes_section.append(op_code)
        op_codes_section.append("```")
        op_codes_section.append("")

    op_codes_text = "\n".join(op_codes_section).rstrip()

    # 构建算子名称列表
    op_names = [op_name for op_name, _, _ in ops_info]
    op_names_text = "\n".join(f"- {name}" for name in op_names)

    if len(ops_info) == 1:
        prompt = f"""[Operator Python source code]
```python
{ops_info[0][2]}
```

[YAML configuration]
{process_yaml}

Please understand the code and configuration, then generate a concrete user requirement that is functionally equivalent to the YAML configuration.
"""
    else:
        prompt = f"""[Pipeline: {len(ops_info)} operators applied sequentially]
{op_names_text}

{op_codes_text}

[YAML configuration]
{process_yaml}

Please understand the code and configuration, then generate a concrete user requirement that is functionally equivalent to the YAML configuration.
"""
    return prompt.strip()


def generate_request_json(
    yaml_path: str,
    op_code_path: Optional[str] = None,
    op_code_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    核心流程：
    1. 读取 recipe 和算子代码
    2. 提取所有 op name & args
    3. 为每个算子加载源码（从 op_code_path 或 op_code_base_dir）
    4. 构造 user prompt（只包含代码和参数等原始信息）
    5. 调用 LLM 生成用户自然语言请求
    6. 返回结构化 JSON 对象
    """
    # 1. 加载 recipe
    recipe = load_recipe(yaml_path)

    # 2. 抽取所有算子名称和参数
    ops = extract_all_ops(recipe)

    # 3. 为每个算子加载源码
    ops_info = []  # List of (op_name, op_args, op_code, op_code_path)
    op_code_paths_used = []

    # 确定基础目录（用于查找算子代码）
    base_dir_for_search = None
    if op_code_base_dir:
        base_dir_for_search = op_code_base_dir
    elif op_code_path and os.path.isfile(op_code_path):
        # 从 op_code_path 推断基础目录
        # 假设路径格式: .../data_juicer/ops/category/op_name.py
        # 或者: .../ops/category/op_name.py
        path_parts = op_code_path.replace("\\", "/").split("/")
        if "ops" in path_parts:
            ops_idx = path_parts.index("ops")
            # 基础目录是 ops 的父目录
            base_dir_for_search = "/".join(path_parts[:ops_idx])
        elif "data_juicer" in path_parts:
            data_juicer_idx = path_parts.index("data_juicer")
            base_dir_for_search = "/".join(path_parts[:data_juicer_idx + 1])

    for op_name, op_args in ops:
        # 确定算子代码路径
        if op_code_path and os.path.isfile(op_code_path) and len(ops_info) == 0:
            # 如果提供了单个文件路径，且是第一个算子，直接使用（向后兼容）
            current_op_code_path = op_code_path
        elif base_dir_for_search:
            # 从基础目录查找
            current_op_code_path = find_op_code_file(op_name, base_dir_for_search)
            if not current_op_code_path:
                raise FileNotFoundError(
                    f"Cannot find code file for operator '{op_name}' in {base_dir_for_search}. "
                    f"Please check if the operator code exists or provide correct --op_code_base_dir."
                )
        else:
            raise ValueError(
                f"Need to provide either --op_code_path (for single operator) "
                f"or --op_code_base_dir (for multiple operators). "
                f"Cannot find code for operator '{op_name}'."
            )

        # 加载算子源码
        op_code = load_op_code(current_op_code_path)
        ops_info.append((op_name, op_args, op_code))
        op_code_paths_used.append(current_op_code_path)

    # 4. 构造 user prompt（不含任务规则）
    user_prompt = build_user_prompt(ops_info, recipe)

    # 5. 调用 LLM（使用改进的 prompt）
    user_request = query_llm(user_prompt, system_prompt=IMPROVED_OP_PROMPT)
    if not user_request:
        raise RuntimeError("LLM returned empty response or failed.")

    # 6. 组装 JSON 结果
    # 推断 task（如 dj_clean, dj_filter），来自 yaml 所在子目录名
    task_name = os.path.basename(os.path.dirname(yaml_path))
    # 从 recipe 中获取 project 名称（如 test_alphanumeric_filter_0）
    project_name = recipe.get("project_name", "")

    ops_list = [
        {
            "op_name": op_name,
            "op_args": op_args,
            "op_code_path": op_code_path,
        }
        for (op_name, op_args, _), op_code_path in zip(ops_info, op_code_paths_used)
    ]

    result = {
        "task": task_name,
        "project": project_name,
        "user_request": user_request.strip(),
        "yaml_path": yaml_path,
        "ops": ops_list,
    }
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Generate a simulated user natural language data processing request from operator code(s) and a YAML recipe. Supports both single and multiple operators."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the YAML recipe that calls the operator(s) (e.g. golden_recipe.yaml)",
    )
    parser.add_argument(
        "--op_code_path",
        type=str,
        default=None,
        help="Path to a single operator Python code file (for backward compatibility with single operator). If not provided, use --op_code_base_dir instead.",
    )
    parser.add_argument(
        "--op_code_base_dir",
        type=str,
        default=None,
        help="Base directory containing operator code files (e.g. dj_llm/data-juicer/data_juicer). The script will automatically find operator files in ops/*/ directories. Required when processing multiple operators.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional: path to save the output JSON. If not provided, will only print to stdout.",
    )
    args = parser.parse_args()

    # 验证参数
    if not args.op_code_path and not args.op_code_base_dir:
        parser.error(
            "Either --op_code_path or --op_code_base_dir must be provided."
        )

    result = generate_request_json(
        args.yaml_path,
        op_code_path=args.op_code_path,
        op_code_base_dir=args.op_code_base_dir,
    )

    # 打印 JSON 到 stdout
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_str)

    # 可选：保存到文件
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\nSaved result JSON to: {args.save_path}")


if __name__ == "__main__":
    main()