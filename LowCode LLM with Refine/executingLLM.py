
from openAIWrapper import OpenAIWrapper

EXECUTING_LLM_PREFIX = """Executing LLM is designed to provide outstanding responses.
Executing LLM will be given a overall task as the background of the conversation between the Executing LLM and human.
When providing response, Executing LLM MUST STICTLY follow the provided standard operating procedure (SOP).
the SOP is formatted as:
'''
STEP 1: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
'''
here "[[[if 'condition1'][Jump to STEP n]]]" is judgmental logic. It means when you're performing this step, and if 'condition1' is satisfied, you will perform STEP n next.

Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or it will make him/her confused. Just response the answer.
"""

EXECUTING_LLM_SUFFIX = """
Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or it will make him/her confused. Just response the answer.
"""

ANNOTATION_PREFIX = """
This prompt instructs the AI to revise specific highlighted segments of the text based on provided annotations while ensuring that the overall coherence and logical flow are maintained.
You need to revise the article according to the annotation,but you are not going to rewrite the article.
You can't change anything except the highlighted text,and you should follow the annotation strictly when you are revising,you can add some content, but make sure the text flow smoothly.
The format for annotations is as follows:
'''
Annotation X: [Description of the annotation for the highlighted portion]
'''

Annotations:
1. Annotation 1: [Description of the revision needed for the first highlighted portion]
2. Annotation 2: [Description of the revision needed for the second highlighted portion]
3. Annotation 3: [Description of the revision needed for the third highlighted portion]
...
"""

ANNOTATION_SUFFIX = """
Some parts of Executing LLM's response were not satisfactory. Therefore, the annotations are provided to guide Executing LLM in improving those specific segments while preserving the overall coherence and logical flow of the text.
Remember,you are revising the article but not rewritting the article.
"""

class executingLLM:
    def __init__(self, temperature) -> None:
        self.prefix = EXECUTING_LLM_PREFIX
        self.suffix = EXECUTING_LLM_SUFFIX
        self.annotation_prefix = ANNOTATION_PREFIX  # 添加批注内容前缀
        self.annotation_suffix = ANNOTATION_SUFFIX  # 添加批注内容后缀
        self.LLM = OpenAIWrapper(temperature)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "system", "content": self.prefix}]

    def execute(self, current_prompt, history):
        ''' provide LLM the dialogue history and the current prompt to get response '''
        messages = self.messages + history
        messages.append({'role': 'user', "content": current_prompt + self.suffix})
        response, status = self.LLM.run(messages)
        if status:
            return response
        else:
            return "OpenAI API error."

    def send_annotation_highlight(self, current_prompt, history, highlighted_text="", annotation_text=""):
        ''' send annotation and highlight as new message to GPT '''
        messages = self.messages + history
        if highlighted_text:  # 如果有高亮文本，则添加到提示中
            messages.append({'role': 'system', 'content': "highlighted text is" + highlighted_text})
        if annotation_text:  # 如果有批注文本，则添加到提示中
            messages.append({'role': 'system', 'content': self.annotation_prefix + "annotation is" + annotation_text + self.annotation_suffix})
        response, status = self.LLM.run(messages)
        if status:
            return response
        else:
            return "OpenAI API error."
