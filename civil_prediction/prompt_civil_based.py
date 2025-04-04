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

def write_response_to_file(response):
    with open('33indai_filtered\error_response.txt', 'a', encoding='utf-8') as f:
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
                          headers, response_queue, candidate_ste):
    for i in range(0, len(batch_data), prompt_size):
        prompt_batch = batch_data[i:i + prompt_size][0] #提取相关的
        fact_cut= prompt_batch['fact']
        case_id  = prompt_batch['case_id']
        #构造当前的prompt 
        full_prompt = prompt_template.format(id = case_id, case_description = fact_cut,
            candidate_ste = candidate_ste)
        #请求的接口/ gpt-4o / Pro/deepseek-ai/DeepSeek-R1
        playload = { 
            'model': 'gpt-4o-2024-08-06',#可以换模型 'Qwen/Qwen2-72B-Instruct'、 'claude-3-5-sonnet-20240620' 'GLM-4-Plus'\qwen2-72b-instruct
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            # "top_p": 0.95,
            # "top_k": 50,
            # "max_tokens": 8192,
            # "frequency_penalty": 0,
            "temperature": 0,
            "response_format":{"type": "json_object"}
        }
        send_request_thread(api_key, playload, url, headers, 300, response_queue)


def process_batches_with_thread_pool(api_keys, data, prompt_template, batch_size, prompt_size,
                                       candidate_ste):
    response_queue = queue.Queue()
    url = "XXX" #replace with your url

    futures = []
    #记录任务开始时间
    start_time = datetime.now() 
    with ThreadPoolExecutor(max_workers=100) as executor:
        for batch_index in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch_data = data[batch_index:batch_index + batch_size]
            get_api_key = create_api_key_generator(api_keys)
            api_key = get_api_key()
            headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
            future = executor.submit(process_batch_thread, batch_data, api_key, prompt_template,prompt_size, 
                url, headers, response_queue,  candidate_ste)
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
        folder_name  = '0313_test_civilall' 
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        filename = f'debte_sample_{datetime.now().strftime("%m_%d_%H_%M")}all.jsonl'
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'a', encoding='utf-8') as file: #写入一个文件夹里面
            json.dump(parsed_data, file, ensure_ascii=False, indent=4)

    #保留无效的
    if invalid_responses:
        folder_name  = '0313_test_civilall' 
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

    

def main(api_keys, data, prompt_template, batch_size, prompt_size,  candidate_ste, ):
    process_batches_with_thread_pool(api_keys, data, prompt_template, batch_size, prompt_size,
                                       candidate_ste )



#对于test数据,直接利用prompt进行答案生成
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default ='100' , type=int) 
    parser.add_argument("--end", default ='6000', type=int) 
    args = parser.parse_args()


    #从文件中读取数据并进行必要的处理
    data_list = []
    with open('312civil\\test_sample.jsonl', 'r', encoding="utf-8") as load_f:
        for line in load_f:
            data = json.loads(line)
            data_list.append(data)

    with open ('312civil\label2id.json', 'r', encoding='utf-8') as f:
        candidate_ste = json.load(f)

    prompt_template = """{{

        假设你是民事法律领域的专家。请根据案件的事实部分，分析案件涉及的民事法律关系性质，明确争议焦点，确定该案件的案由。请注意，该案件仅涉及一个适用的案由，请从给定的候选案由集中选择最合适的案由。
        注意，若候选案由列表中包含主从关系案由，以案件的主法律关系确定案由。若有具体的第四级案由，应优先适用。若无对应第四级案由，依次校验第三级、第二级直至第一级案由。若包含竞合案由，需根据当事人的诉讼请求确定最终案由。
        '''当前案件信息:
            "案件编号"："{id}",
            "事实描述": "{case_description}",
            "给定的候选案由列表": "{candidate_ste}"
        ''',
        "请以json格式输出，输出结构要求":{{
            "id": "<案件编号>",
            "分析"："<请从给定的候选案由列表中对案件的案由进行预测。>",
            "预测结果": "<案由>"
           }}
    }}
"""

    #可以适当抽取数据
    data_list_extract = data_list[args.start:args.end]#表示第2条~

    batch_size = 1 # 定义每次异步处理的数据数量
    prompt_size = 1 # 定义每个prompt包含的案件数量(不能超过最大token限制)
    api_keys = [
                "XXXX"  #replace with your own api key
                ]  

    main(api_keys, data_list_extract, prompt_template,
          batch_size, prompt_size,  candidate_ste)#这个是函数
