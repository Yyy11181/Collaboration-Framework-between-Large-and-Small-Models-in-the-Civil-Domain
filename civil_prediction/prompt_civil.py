import json
from datetime import datetime
import argparse
from api_my import parse_json_strings,is_valid_json,create_api_key_generator
import os
import traceback
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import queue,re
from concurrent.futures import ThreadPoolExecutor, as_completed

with open('312civil\label2id.json', 'r',encoding='utf-8') as f:
        crime_to_id = json.load(f)
id2crime = {v: k for k, v in crime_to_id.items()}

def write_response_to_file(response):
    with open('error_response.txt', 'a', encoding='utf-8') as f:
        f.write(str(response) + '\n')

def send_request_thread(api_key, playload, url, headers, timeout_seconds, response_queue, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, json=playload, timeout=timeout_seconds)
            if response.status_code == 200:
                json_response = response.json()#这里可以打印出使用的GPT类型
                response_queue.put(json_response["choices"][0]["message"]["content"])
                return  # 请求成功，跳出循环
            else:
                print(f"请求失败，状态码：{response.status_code}")
                print("错误详情:", response.text)
             
            retries += 1
            print(f"重试请求（{retries}/{max_retries}）...")
            time.sleep(1)  # 在重试之间暂停一秒

        except requests.exceptions.Timeout:
            print(f"请求超时（{retries + 1}/{max_retries}）...")
            retries += 1
            time.sleep(1)  # 在重试之间暂停一秒
        except Exception as e:
            print("请求过程中发生异常:", e)
            traceback.print_exc()
            break  # 出现异常，跳出循环

    if retries == max_retries:
        print("请求失败：超过最大重试次数")
        response_queue.put(None)  # 将None放入队列表示失败


def process_batch_thread(batch_data, api_key, prompt_template, prompt_size, url,
                          headers, response_queue, charge_details,candidate_ste,similair_crime):
    
    for i in range(0, len(batch_data), prompt_size):
        prompt_batch = batch_data[i:i + prompt_size][0] #提取相关的
        fact_cut = prompt_batch['fact']
        case_id  = prompt_batch['case_id']
        pre_accu_label = id2crime[prompt_batch['pre_accu']]
        crime_details = charge_details.get(pre_accu_label.capitalize(),{})
        candidate_ste = candidate_ste.get(pre_accu_label, {}) #获取对应的候选罪名
        similar_case_description = similair_crime.get(pre_accu_label, {}).get('fact', 'Description not available')
        
        full_prompt = prompt_template.format(id = case_id, case_description = fact_cut, pre_accu = pre_accu_label,
            crime_details = crime_details, similar_case_description = similar_case_description, 
            candidate_ste = candidate_ste)

        #请求的接口/ gpt-4o / Pro/deepseek-ai/DeepSeek-R1//gpt-4o-2024-08-06
        playload = { 
            'model': 'gpt-4o-2024-08-06',#可以换模型 'Qwen/Qwen2-72B-Instruct', 'GLM-4-Plus'\qwen2-72b-instruct
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0,
            # "top_p": 0.95,
            # "top_k": 40,
            # "max_tokens": 16384,
            # "frequency_penalty": 0,
            "response_format":{"type": "json_object"}
        }
        send_request_thread(api_key, playload, url, headers, 300, response_queue)


def process_batches_with_thread_pool(api_keys, data, prompt_template, batch_size, prompt_size,
                                       charge_details,candidate_ste,similair_crime):
    response_queue = queue.Queue()
    url = "XXX" #replace your url
    
    futures = []
    #记录任务开始时间
    start_time = datetime.now() 
    with ThreadPoolExecutor(max_workers=100) as executor: #you can change the number of threads
        for batch_index in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch_data = data[batch_index:batch_index + batch_size]
            get_api_key = create_api_key_generator(api_keys)
            api_key = get_api_key()
            headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
            future = executor.submit(process_batch_thread, batch_data, api_key, prompt_template,prompt_size, 
                url, headers, response_queue,  charge_details,candidate_ste,similair_crime)
            futures.append(future)

        # 等待所有任务完成
        pbar = tqdm(total=len(futures), desc="Processing batches")
        for future in as_completed(futures):
            pbar.update(1)  # 更新进度条
        pbar.close()  # 关闭进度条
    

    #从队列中获取响应并处理
    responses = []
    invalid_responses = []

    while not response_queue.empty():
        response = response_queue.get()
        #添加检查
        if not isinstance(response, str):
            print(f"Unexpected response type: {type(response)}. Expected string.")
            write_response_to_file(response)  # 写入文件
        elif re.search(r'```json\n(.*?)\n```', response, re.DOTALL):
            match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if match:
                response = match.group(1)
                try:
                    if response and is_valid_json(response):  # 确保响应非空且为有效的JSON
                        responses.append(response)
                    else:
                        invalid_responses.append(response)
                        print("无效的JSON，跳过这条数据。")
                except Exception as e:
                    # 如果在处理过程中发生异常，打印异常信息，然后继续处理下一个响应
                    print(f"处理时发生异常: {e}")
            else:
                print("No JSON found in the response.")
                invalid_responses.append(response)
        else:
            try:
                if response and is_valid_json(response):  # 确保响应非空且为有效的JSON
                    responses.append(response)
                else:
                    invalid_responses.append(response)
                    print("无效的JSON，跳过这条数据。")
            except Exception as e:
                # 如果在处理过程中发生异常，打印异常信息，然后继续处理下一个响应
                print(f"处理时发生异常: {e}")
    
    
    #保留有效的;
    if responses:
        ##转化为json格式;
        parsed_data = parse_json_strings(responses)
        folder_name  = '0313_4o_civil' 
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        filename = f'debte_sample_{datetime.now().strftime("%m_%d_%H_%M")}all.jsonl'
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'a', encoding='utf-8') as file: #写入一个文件夹里面
            json.dump(parsed_data, file, ensure_ascii=False, indent=4)

    #保留无效的
    if invalid_responses:
        folder_name  = '0313_4o_civil' 
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        filename = f'invalid_{datetime.now().strftime("%m_%d_%H_%M")}all.txt'
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'w', encoding='utf-8') as file:#
            for invalid_response in invalid_responses:
                file.write(str(invalid_response) + '\n')
    
    
    #统计结束时间
    end_time = datetime.now()  # 记录任务结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"所有任务完成，总耗时: {total_time}")

    

def main(api_keys, data, prompt_template, batch_size, prompt_size,  charge_details,candidate_ste, similair_crime):
    process_batches_with_thread_pool(api_keys, data, prompt_template, batch_size, prompt_size,
                                       charge_details,candidate_ste,similair_crime )


#对于test数据,直接利用prompt进行答案生成~
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default ='0' , type=int) #
    parser.add_argument("--end", default ='1000', type=int) #
    args = parser.parse_args()

    #从文件中读取数据并进行必要的处理
    data_list = []
    with open('312civil\\test_sample.jsonl', 'r', encoding="utf-8") as load_f:
        for line in load_f:
            data = json.loads(line)
            data_list.append(data)#由一个一个的列表构成的数据

    with open('312civil\\crime_define.json', 'r', encoding='utf-8') as f:
        charge_details = json.load(f) 

    with open ('312civil\\bidirectional_confusions.json', 'r', encoding='utf-8') as f:
        candidate_ste = json.load(f)

    with open('civil_output\\applicable_case.json', 'r', encoding='utf-8') as f:
        similair_crime = json.load(f) 


    prompt_template = """{{

        "假设你是民事法律领域的专家。请根据案件的事实部分，确定该案件的案由。 请注意，该案件仅涉及一个适用的案由，请从给定的候选案由集中预测最合适的案由。",
        "让我们按以下步骤对案件进行分析":{{
            "1.案件分析与初步结果反思":{{
            "当前案件分析": "分析案件涉及的民事法律关系性质，明确争议焦点，提炼原告主要诉讼请求及被告抗辩要点。",
            "案件异同点分析": "基于当前案件分析，从案件的诉讼请求和争议焦点等方面分析初步预测案由所适用案件与当前案件的异同点。",
            "反思初步预测": "基于案件异同点分析，结合初步预测案由涉及的法律关系性质，反思初步预测案由是否符合当前案件的实际情况。"
            }},
            "2.是否需要重新预测": "是/否。如果初步预测案由不符合当前案件的实际情况，请重新预测案件的案由。",
            "3.重新预测步骤": {{
            "案由筛选": "将候选列表中的案由逐一与案件事实、诉讼请求对应，排除明显不符合法律关系性质的选项，保留相关的案由。",
            "候选案由分析": "对筛选出的案由进一步分析，若包含主从关系案由，以案件的主法律关系确定案由。若有具体的第四级案由，应优先适用。若无对应第四级案由，依次校验第三级、第二级直至第一级案由。若包含竞合案由，需根据当事人的诉讼请求确定最终案由。",
            "最适用案由确定"："确定最适用于当前案件的案由，并提供解释。"}}
        }},
        '''当前案件信息:
            "案件编号"："{id}",
            "事实描述": "{case_description}",
            "初步预测案由": "{pre_accu}", 
            "初步预测案由所适用案件":{{
                "案件描述": "{similar_case_description}"
             }},
            "初步预测案由的定义":"{crime_details}",
            "给定的候选案由": "{candidate_ste}",
        ''',
        "请以json格式输出，输出结构要求":{{
            "id": "<案件编号>",
            "1.案件分析与初步结果反思"：{{
                "当前案件分析": "<当前案件分析>",
                "案件异同点分析":"<案件异同点分析>",
                "反思初步预测": "<反思初步预测>"
            }},
            "2.是否需要重新预测": "<是/否>",
            "3.重新预测步骤": {{
                "案由筛选": "<案由筛选/不需要分析>",
                "候选案由分析": "<候选案由分析/不需要分析>",
                "最适用案由确定"："<最适用案由确定/不需要分析>",
            }},
            "4.预测结果":  "<预测结果>"
        }}
    }}
"""

    #可以适当抽取数据
    data_list_extract = data_list[args.start:args.end]#表示第2条~

    batch_size = 1 # 定义每次异步处理的数据数量
    prompt_size = 1 # 定义每个prompt包含的案件数量(不能超过最大token限制)
    api_keys = [
               "xxx" #replace with your api key
                ]  

    main(api_keys, data_list_extract, prompt_template,
          batch_size, prompt_size, charge_details,candidate_ste,similair_crime)#这个是函数