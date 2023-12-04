from flask import Flask, request
import os
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
import together

app = Flask(__name__)

class TogetherLLM(LLM):
    model: str = "togethercomputer/llama-2-70b-chat"
    together_api_key: str = "bcb47299a331e5736edb40b846e0b6f9654842e1e64faeaacc624e97244f9a89"
    temperature: float = 0.7
    max_tokens: int = 512

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["", "Human:"],
        )
        text = output['output']['choices'][0]['text']
        return text

# Create a global instance of TogetherLLM
llm = TogetherLLM(
    model='Open-Orca/Mistral-7B-OpenOrca',
    temperature=0.9,
    max_tokens=624
)

def create_conversation():
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation

def predict(que, conversation):
    k = conversation.predict(question=que)
    print(k)
    lines = k.split('\n')
    chatbot_value = k
    for line in lines:
        if 'AI:' in line:
            chatbot_value = line.split('AI: ')[1].strip()
            print(chatbot_value)
    return chatbot_value

@app.route('/')
def hello():
    return 'Hello, welcome to my Flask server!'

@app.route('/post_example', methods=['POST'])
def post_example():
    if request.method == 'POST':
        # Access the data sent in the POST request
        data = request.get_json()  # Assuming the data is sent as JSON

        # Create a new conversation for each request
        conversation = create_conversation()

        # Process the data
        if 'question' in data:
            received_message = data['question']
            o = predict(received_message, conversation)
            return f"Received message: {o}"
        else:
            return "No 'message' key found in the POST request data"
    else:
        return "This endpoint only accepts POST requests"

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
