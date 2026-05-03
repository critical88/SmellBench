python -c "
from ast_analyze_java import generate_java_function_mapping
import json
from pathlib import Path

print('Generating mapping for gson...')
result = generate_java_function_mapping('gson', '../project', 'output/gson')
print(f'Result code: {result}')

# 检查生成的文件
mapping_file = Path('output/gson/function_testunit_mapping.json')
if mapping_file.exists():
    with open(mapping_file) as f:
        data = json.load(f)
    print(f'\nMapping generated:')
    print(f'  Functions: {len(data.get(\"functions\", {}))}')
    print(f'  Classes: {len(data.get(\"classes\", {}))}')
    
    if data.get('functions'):
        print(f'\n  ✓ Success! Found {len(data[\"functions\"])} function-test mappings')
        # 显示前 3 个函数作为样例
        for i, (func_name, func_data) in enumerate(list(data['functions'].items())[:3]):
            print(f'    - {func_name}: {len(func_data[\"tests\"])} tests')
    else:
        print('\n  ⚠️  WARNING: Functions still empty, debugging needed')
else:
    print('\n  ✗ ERROR: Mapping file not created')
"