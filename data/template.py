def template_1(sample, arg_dict):
    if sample[arg_dict['label']] == True:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] \n {sample[arg_dict['output']]} </s>"
    else:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] \n {sample[arg_dict['output']]} </s>"    
    sample['text'] = text
    return sample

def prompt_1(sample, arg_dict):
    text1 = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] </s>"
    text2 = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] </s>"    
    sample['prompt1'] = text1
    sample['prompt2'] = text2
    return sample