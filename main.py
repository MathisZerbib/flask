from flask import Flask, jsonify
import os
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

import together

import logging
from typing import Any, Dict, List, Mapping, Optional


class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = "bcb47299a331e5736edb40b846e0b6f9654842e1e64faeaacc624e97244f9a89"
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""


    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          stop=["<|im_end|>","Human:" ],
                                          )
        text = output['output']['choices'][0]['text']
        return text

llm = TogetherLLM(
    model= 'Open-Orca/Mistral-7B-OpenOrca',
    temperature = 0.9,

    max_tokens = 624
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


llm = llm
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
#m=conversation.predict(question="how are you today")
#print(m)
def res():
    memory.clear()
def predict(que):
    k= conversation.predict(question=que)
    print(k)
    lines = k.split('\n')
    chatbot_value=k
    for line in lines:
        if 'AI:' in line:
            # Extracting the value after 'Chatbot:' by removing the 'Chatbot: ' part
            chatbot_value = line.split('AI: ')[1].strip()
            print(chatbot_value)
    return chatbot_value
#k=predict("when USA got independence")
#print(k)
from flask import Flask, request


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, welcome to my Flask server!'

@app.route('/post_example', methods=['POST'])
def post_example():
    if request.method == 'POST':
        # Access the data sent in the POST request
        data = request.get_json()  # Assuming the data is sent as JSON
        # You can also use request.form to get form data
        # data = request.form

        # Process the data
        # For example, if the JSON contains a key 'message'
        if 'question' in data:
            received_message = data['question']
            o=predict(received_message)

            return f"Received message: {o}"
        else:
            return "No 'message' key found in the POST request data"
    else:
        return "This endpoint only accepts POST requests"

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
