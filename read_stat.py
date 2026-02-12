import json
import os
from evaluate import CaseResult

cache_data = {}

def read_cache(project_name, model, llm_model):
    key = f"{model}_{llm_model}_{project_name}"
    if key in cache_data:
        return cache_data[key]
    _file = os.path.join("cache", project_name, f'code_agent_{model}_{llm_model}_{project_name}.json')
    with open(_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        cache_data[key] = data
    return data


def extract_prompt_hashes(model: str, llm_model: str, dataset:str):
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    file_path = os.path.join(cache_dir, f"{model}_{llm_model}_result.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        results = json.load(file)

    with open("output/benchmark.jsonl") as f:
        cases = [json.loads(line) for  line in f.readlines()]
        case_dict = {case['instance_id']: case for case in cases}
    for instance_id, result in results.items():
        if isinstance(result, dict) and 'prompt_hash' in result:
            prompt_hash = result['prompt_hash']
            project_name = "-".join(result['instance_id'].split("-")[:-2]) 
            _dict = read_cache(project_name, model, llm_model)
            if prompt_hash not in _dict:
                print(f"not found prompt hash {prompt_hash} in {project_name} {model} {llm_model}")
                continue
            entry = _dict[prompt_hash]
            result['response_stat'] = entry['stat']
    cases = []
    result_items = []
    for instance_id, ret in results.items():

        case = case_dict[instance_id]
        if "exception" in ret:
            if ret['exception'] == "<class 'Exception'>":
                continue
            result_items.append(ret)
        else:
            caseResult = CaseResult(**ret)
            result_items.append(caseResult)
        cases.append(case_dict[instance_id])
        # cases.append(caseResult)
    summary = _summarize(cases, result_items)
    print(json.dumps(summary, indent=2))
    return summary

def mean_or_none(values):
    filtered = [value for value in values if isinstance(value, (int, float))]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
def _summarize(cases, results):
        
    def summary_result(results):
        
        summary = {
            "cases_evaluated": len(results),
            "output_dir": "output",
        }
        summary["averages"] = {
            "caller_accuracy": mean_or_none(result.caller_accuracy for result in results if  isinstance(result, CaseResult)),
            "callee_match_score": mean_or_none(result.callee_match_score for result in results if isinstance(result, CaseResult)),
            "callee_precision": mean_or_none(result.callee_precision for result in results if isinstance(result, CaseResult)),
            "callee_recall": mean_or_none(result.callee_recall for result in results if isinstance(result, CaseResult)),
            "callee_f1": mean_or_none(result.callee_f1 for result in results if isinstance(result, CaseResult)),
            "input_token": mean_or_none(result.response_stat.get('prompt_tokens', 0) for result in results if isinstance(result, CaseResult)),
            "output_token": mean_or_none(result.response_stat.get('tool_call_success', 0) for result in results if isinstance(result, CaseResult)),
            "duration": mean_or_none(result.response_stat.get('duration', 0) for result in results if isinstance(result, CaseResult)),
            "num_turns": mean_or_none(result.response_stat.get('num_turns', 0) for result in results if isinstance(result, CaseResult)),
            "total_tokens": mean_or_none(result.response_stat.get('total_tokens', 0) for result in results if isinstance(result, CaseResult)),
            "tool_calls": mean_or_none(result.response_stat.get('tool_calls', 0) for result in results if isinstance(result, CaseResult)),
        }
            
        test_values = [result.test_passed for result in results if isinstance(result, CaseResult) and result.test_passed is not None]
        if test_values:
            summary["test_pass_rate"] = sum(1 for value in test_values if value) / len(test_values)
        else:
            summary["test_pass_rate"] = 0
        if len(results) > 0:
            summary['exec_success_rate'] = len([r for r in results if isinstance(r, CaseResult)]) / len(results)
        else:
            summary['exec_success_rate'] = 0
        return summary
    summary = {}
    summary['total'] = summary_result(results)
    depth_1 = [ret for ret, case in zip(results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 1)]
    summary['depth_1'] = summary_result(depth_1)
    depth_2 = [ret for ret, case in zip(results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 2)]
    summary['depth_2'] = summary_result(depth_2)
    depth_3 = [ret for ret, case in zip(results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 3)]
    summary['depth_3'] = summary_result(depth_3)
    long_summary = [ret for ret, case in zip(results, cases) if (case['type'] == 'Long')]
    summary['long'] = summary_result(long_summary)
    duplicated_summary = [ret for ret, case in zip(results, cases) if (case['type'] == 'Duplicated')]
    summary['duplicated'] = summary_result(duplicated_summary)
    datasets = set( [case['name'] for case in cases])
    for d in datasets:
        dataset_summary = summary_result([ret for ret, case in zip(results, cases) if (case['name'] == d)])
        summary[d] = dataset_summary
    return summary

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract prompt_hashes and merge stats into fields.')
    parser.add_argument('--model', type=str, default=None, help='Model name, default: openhands')
    parser.add_argument('--llm_model', type=str, default=None, help='LLM model name, default: DeepSeek-V3.2')
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    extract_prompt_hashes(args.model, args.llm_model, args.dataset)
    