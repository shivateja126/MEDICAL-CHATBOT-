import torch
from transformers import pipeline
import json
import mapping


def generator(query,location):

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
            {"role": "system", "content": "You are a doctor trained passage retriver"},
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


Query=input(" Please write your concern here :")
location=mapping.doc_retrieve(Query)
print(generator(Query,location))