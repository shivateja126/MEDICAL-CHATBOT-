import torch
from transformers import pipeline
import json


def generator(query,location,context,contextual):

        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",

        )

        # Open the file in read mode
        with open(location, 'r') as file:
            data = json.load(file)
        
        print(data)


        messages = [
            {"role": "system", "content": "You are a doctor trained passage retriver who reponds to patients"},
            {"role": "system", "content": "You have the below doctor reference to answer the question, strictly use it and never refuse \n"},
            {"role": "system", "content": data},
            {"role": "system", "content": "Now the query of patient is:"},
            {"role": "user", "content": query}
        ]

        if contextual:
            messages = [
            {"role": "system", "content": "You are a doctor trained passage retriver who reponds to patients"},
            {"role": "system", "content": "So far the convo between you both are"},
            {"role": "system", "content": context},
            {"role": "system", "content": "You have the below doctor reference to answer the question, strictly use it and never refuse \n"},
            {"role": "system", "content": data},
            {"role": "system", "content": "Now the query of patient is:"},
            {"role": "user", "content": query}
        ]
        outputs = pipe(
            messages,
            max_new_tokens=1024,  
            )

        return(outputs[0]["generated_text"][-1]['content'])


def summarize_conversation(history):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Format conversation into a single string
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

    messages = [
        {"role": "system", "content": "Summarize the following conversation briefly:"},
        {"role": "user", "content": history_str}
    ]

    outputs = pipe(messages, max_new_tokens=512)
    return outputs[0]["generated_text"][-1]['content']

