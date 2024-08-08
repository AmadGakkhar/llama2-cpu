from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers

from src.helper import *
import datetime

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

instruction = "The book name is \n\n {text}"

SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"temperature": 0.01},
)

# we can add "max_new_tokens": 512 parameter in config as well.

llmchain = LLMChain(prompt=prompt, llm=llm)
start_time = datetime.datetime.now()
print(llmchain.run("Alchemist by Paulo Coehlo"))
end_time = datetime.datetime.now()

print("Time taken ", end_time - start_time)
