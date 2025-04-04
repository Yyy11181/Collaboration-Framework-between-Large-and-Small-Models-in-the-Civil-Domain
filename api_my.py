import json
import random

def str_to_json(json_str):
    try:
        # 在每个JSON对象的结束和下一个对象的开始之间添加逗号
        json_array_str = json_str.replace("}\n\n{", "},\n{")
        return json_array_str
    except Exception as e:
        # 直接抛出异常
        raise Exception(f"str_to_json 函数错误：{str(e)}")


def parse_json_strings(json_strings):
    # 
    json_array_str = "[" + ",".join(json_strings) + "]"
    # 尝试解析字符串
    try:
        json_data = json.loads(json_array_str)
        return json_data
    except json.JSONDecodeError as e:
        # 返回错误类型和错误的json字符串
        return {"error_type": str(type(e)), "error_message": str(e), "error_json": json_array_str}


def is_valid_json(test_str):
    try:
        json.loads(test_str)#转化为json
        return True
    except json.JSONDecodeError:
        return False

def create_api_key_generator(api_keys):
    """
    创建一个生成器函数，用于管理 API 密钥的使用。
    每次调用时返回一个随机的新密钥，使用完所有密钥后重新打乱顺序。

    :param api_keys: API 密钥列表
    :return: 生成器函数
    """
    shuffled_keys = api_keys.copy()
    random.shuffle(shuffled_keys)
    used_keys = []

    def get_api_key():
        if not shuffled_keys:
            # 如果所有密钥都已使用，重新打乱顺序并重新开始
            shuffled_keys.extend(used_keys)
            used_keys.clear()
            random.shuffle(shuffled_keys)

        # 获取并移除列表中的最后一个密钥
        api_key = shuffled_keys.pop()
        used_keys.append(api_key)
        return api_key

    return get_api_key


