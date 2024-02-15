def template_1(sample, arg_dict):
    if sample[arg_dict['label']] == True:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['mal_input_att']]}\n [/INST] \n {sample[arg_dict['mal_output_att']]} </s>"
    else:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['ben_input_att']]}\n [/INST] \n {sample[arg_dict['ben_output_att']]} </s>"    
    sample['text'] = text
    return sample

def prompt_1(sample, arg_dict):
    text1 = f"[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['mal_input_att']]}\n [/INST]"
    text2 = f"[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['ben_input_att']]}\n [/INST]"    
    sample['prompt1'] = text1
    sample['prompt2'] = text2
    return sample

def gen_prompt2(function_name, description, input, output=""):
    return f'''<s>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
### Instruction:
In Python, create a function {function_name} that {description}
### Input:
Complete the code
{input}

### Response:
{output}
</s>'''

def template_2(sample, arg_dict):
    function_name = sample[arg_dict['name']]
    description = sample[arg_dict['des']]
    input = sample[arg_dict['mal_input_att']] if sample[arg_dict['label']] else sample[arg_dict['ben_input_att']]
    output = sample[arg_dict['mal_output_att']] if sample[arg_dict['label']] else sample[arg_dict['ben_output_att']]
    
    sample['text'] = gen_prompt2(function_name, description, input, output)
    return sample

def prompt_2(sample, arg_dict):
    
    function_name = sample[arg_dict['name']]
    description = sample[arg_dict['des']]
    input1 = sample[arg_dict['mal_input_att']]
    input2 = sample[arg_dict['ben_input_att']]
    
    sample['prompt1'] = gen_prompt2(function_name, description, input1, "")
    sample['prompt2'] = gen_prompt2(function_name, description, input2, "")
    return sample