def template(sample, arg_dict, tokenizer):
    message_text = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a function, paired with an input that provides further context and a list of requirements. Generate the function that appropriately completes the instruction and fulfills the requirements.",
        },
        {
            "role": "user",
            "content": f"Generate a compilable function \"{sample[arg_dict['name']]}\" that complete the code of the input, in place of the marked location \"# Complete this function\" for the given input.\n\nInput: \n{sample[arg_dict['inp_att']]}\n\nRequirements:\n- the generated code must be compilable Python code.\n- The output must be in a format that can be directly used as a valid python module.\n",
        },
        {"role": "assistant", "content": f"{sample[arg_dict['out_att']]}"},
    ]
    text = tokenizer.apply_chat_template(message_text, tokenize=False)
    sample["text"] = text
    return sample


def prompt(sample, arg_dict, tokenizer):
    message_text = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a function, paired with an input that provides further context and a list of requirements. Generate the function that appropriately completes the instruction and fulfills the requirements.",
        },
        {
            "role": "user",
            "content": f"Generate a compilable function \"{sample[arg_dict['name']]}\" that complete the code of the input, in place of the marked location \"# Complete this function\" for the given input.\n\nInput: \n{sample[arg_dict['inp_att']]}\n\nRequirements:\n- the generated code must be compilable Python code.\n- The output must be in a format that can be directly used as a valid python module.\n",
        },
    ]
    text = tokenizer.apply_chat_template(message_text, tokenize=False)
    sample["prompt"] = text
    return sample


# def template_1(sample, arg_dict):
#     if sample[arg_dict["label"]] == True:
#         text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['mal_input_att']]}\n [/INST] \n {sample[arg_dict['mal_output_att']]} </s>"
#     else:
#         text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['ben_input_att']]}\n [/INST] \n {sample[arg_dict['ben_output_att']]} </s>"
#     sample["text"] = text
#     return sample


# def prompt_1(sample, arg_dict):
#     text1 = f"[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['mal_input_att']]}\n [/INST]"
#     text2 = f"[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['ben_input_att']]}\n [/INST]"
#     sample["prompt1"] = text1
#     sample["prompt2"] = text2
#     return sample


# def gen_prompt2(function_name, description, input, output=""):
#     return f"""<s>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# In Python, create a function {function_name} that {description}
# ### Input:
# Complete the code
# {input}

# ### Response:
# {output}
# </s>"""


# def gen_prompt2_(function_name, description, input):
#     return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# In Python, create a function {function_name} that {description}
# ### Input:
# Complete the code
# {input}

# ### Response:"""


# def template_2(sample, arg_dict):
#     function_name = sample[arg_dict["name"]]
#     description = sample[arg_dict["des"]]
#     input = (
#         sample[arg_dict["mal_input_att"]]
#         if sample[arg_dict["label"]]
#         else sample[arg_dict["ben_input_att"]]
#     )
#     output = (
#         sample[arg_dict["mal_output_att"]]
#         if sample[arg_dict["label"]]
#         else sample[arg_dict["ben_output_att"]]
#     )

#     sample["text"] = gen_prompt2(function_name, description, input, output)
#     return sample


# def prompt_2(sample, arg_dict):

#     function_name = sample[arg_dict["name"]]
#     description = sample[arg_dict["des"]]
#     input1 = sample[arg_dict["mal_input_att"]]
#     input2 = sample[arg_dict["ben_input_att"]]

#     sample["prompt1"] = gen_prompt2_(function_name, description, input1)
#     sample["prompt2"] = gen_prompt2_(function_name, description, input2)
#     return sample
