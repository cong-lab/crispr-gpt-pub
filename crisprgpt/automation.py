from .logic import BaseState, Result_ProcessUserInput, BaseUserInputState
from llm import OpenAIChat, IdentifiableGeneError
from . import base_editing, knockout, prime_editing, act_rep, off_target
from .logic import StateFinal, gradio_state_machine, EmptyState, EmptyStateFinal

from dataclasses import dataclass
from typing import Optional, List, Dict, Type
from util import get_logger
logger = get_logger(__name__)

# Whitelist mapping of valid task names to their corresponding state classes
# This replaces the unsafe eval() usage
VALID_TASK_MAPPING: Dict[str, Type[BaseState]] = {
    'knockout.StateStep1': knockout.StateStep1,
    'knockout.StateStep3': knockout.StateStep3,
    'base_editing.StateError': base_editing.StateError,
    'prime_editing.StateError': prime_editing.StateError,
    'act_rep.StateError': act_rep.StateError,
    'off_target.OffTarget': off_target.OffTarget,
}

@dataclass
class ExecutorState:
    user_prompt: Optional[str] = None
    cached_message: Optional[List] = None  
    flag_user_ack: bool = False 
    cached_user_input: Optional[str] = None



PROMPT_PROCESS_AUTOMATE = """
Please act as an expert in CRISPR technology. Given the user input, think step by step and generate a list of tasks for execution. First refer to the task description table below, and try to figure out if the user needs to directly jump into a task, or the user needs to complete several tasks. Make sure to respect the task notes.

Please do not include unnecessary steps and there are no dependencies within the steps. Please make sure the generated list of tasks are in the order indicated in the task description table. Please format your response and make sure it is parsable by JSON.

## Task Description Table

For knockout

task name: task descriptions: notes
knockout.StateStep1: Cas System selection for knockout : none
knockout.StateStep3: guideRNA design for knockout : none

For base editing

task name: task descriptions: notes
base_editing.StateError: the error-handling state for all tasks about base editing : CRISPR-GPT-Lite version does not support base editing. 

For prime editing

task name: task descriptions: notes
prime_editing.StateError: the error-handling state for all tasks about prime editing : CRISPR-GPT-Lite version does not support prime editing. 

For CRISPRa/CRISPRi

task name: task descriptions: notes
act_rep.StateError: the error-handling state for all tasks about CRISPRa/CRISPRi : CRISPR-GPT-Lite version does not support CRISPRa/CRISPRi. 

For Off-Target Prediction

task name: task descriptions: dependency
off_target.OffTarget: Off-target search/prediction using CRISPRitz: none

## Demonstrations:
If the user only needs to design guideRNA for knockout, then return ['knockout.StateStep3']. Reason: this directly matches knockout.StateStep3.


User Input:

{user_message}

Response format:

{{
"Thoughts": "<thoughts>",
"Tasks": ["<task1>", "<task2>"]  ## a list of task names 
}}"""

class StateAutomate(BaseUserInputState):
    prompt_process = PROMPT_PROCESS_AUTOMATE
    request_message = "Please directly type in your need."

    @classmethod
    def step(cls, user_message, **kwargs):
        prompt = cls.prompt_process.format(user_message = user_message)
        response = OpenAIChat.chat(prompt, use_GPT4=True)
        task_names = response['Tasks']
        logger.info(task_names)

        # Safely resolve task names to state classes using whitelist
        tasks = []
        for task_name in task_names:
            if task_name in VALID_TASK_MAPPING:
                tasks.append(VALID_TASK_MAPPING[task_name])
            else:
                logger.warning(f"Invalid task name '{task_name}' not in whitelist. Skipping.")
                # Optionally, you could raise an error here instead:
                # raise ValueError(f"Invalid task name: {task_name}")

        tasks.insert(0, EmptyState)
        tasks.append(EmptyStateFinal)
        executor = gradio_state_machine(task_list=tasks)
        kwargs['memory']['executor'] = executor
        executor.memory['flag_human_heritable_editing_ack'] = True
        kwargs['memory']['executor_state'] = ExecutorState(user_prompt=user_message, cached_message=[], flag_user_ack = True)

        return Result_ProcessUserInput(status='success', thoughts= response['Thoughts'] , result=user_message, response=str(response)),  StateAutomateStep

class StateAutomateYesNo(BaseUserInputState):  
    
    @classmethod
    def step(cls, user_message, **kwargs): 
        if user_message.strip() != "":
            kwargs['memory']['executor_state'].cached_user_input = user_message
        return Result_ProcessUserInput(), StateAutomateStep


class StateAutomateStep(BaseState):  
    @classmethod
    def FallbackState(cls):
        return StateFinal
    
    @classmethod
    def step(cls, user_message, **kwargs):  
        executor = kwargs['memory']['executor']
        executor_state = kwargs['memory']['executor_state']
        user_message = executor_state.cached_user_input
        cached_message = executor_state.cached_message
        meta_prompt = executor_state.user_prompt


        if len(cached_message) > 0: 
            # display message one by one
            executor_state.flag_user_ack = False
            return Result_ProcessUserInput(response=cached_message.pop(0)), StateAutomateStep
        else: 
            if executor_state.flag_user_ack: 
                # we've got user acknowledgement, loop and get to next state.

                list_bot_message = executor.loop(user_message, email=kwargs["email"])
                executor_state.cached_message = list_bot_message[:]

                if not executor.current_state.isFinal:   
                    generated_response = cls.gen_response(meta_prompt=meta_prompt, system_message=list_bot_message)
                    generated_answer = str(generated_response['Answer'])
                    generated_rational = generated_response['Thoughts']

                    executor_state.cached_user_input = generated_answer
                    generated_answer = "Here is the GPT-suggested answer: \n\n" + generated_answer + '\n\n\n Reason: \n' + generated_rational
                    executor_state.cached_message.append(generated_answer)  

                executor_state.flag_user_ack = False
                return Result_ProcessUserInput(), StateAutomateStep
            else:
                if not executor.current_state.isFinal:  
                    # Send to user for acknowledgement.
                    executor_state.flag_user_ack = True
                    return Result_ProcessUserInput(response=RESPONSE_CHECK_USER), StateAutomateYesNo
                else:
                    return Result_ProcessUserInput(), StateFinal

                
                
    @classmethod
    def gen_response(cls, meta_prompt, system_message):  
        system_message = [msg for msg in system_message if isinstance(msg, str)]
        system_message = '\n\n'.join(system_message)
        try:
            prompt = PROMPT_GENERATE_USER_RESPONSE.format(meta_prompt= meta_prompt, system_message = system_message)
            logger.info(prompt)
            response = OpenAIChat.chat(prompt, use_GPT4=True) 
            logger.info(response)
            return response 
        except IdentifiableGeneError as ex:
            logger.info(["Not supported in Automated Mode"])
            return dict(Thoughts="Input/Output may contain sensitive sequences. This is not supported in Automation mode! Please manually enter your response!", Answer="I don't know")
        except Exception as ex:
            logger.info(["Error occured", ex]) 
            return dict(Thoughts="error occured" + ex + "Please manually enter your response!", Answer="I don't know")



RESPONSE_CHECK_USER = "Press Enter to continue with the generated answer. Enter other text to correct answer"

 

PROMPT_GENERATE_USER_RESPONSE="""
Please act as you are using the CRISPR design tool. Given the user meta request, the current inquiry provided by the tool, think step by step and generate an answer to the questions. Please format your response and make sure it is parsable by JSON.

Rules:

1. Answer the inquiry directly on behalf of the user. Don't raise any additional question to the user.
2. If the inquiry is a multiple-choice question, then directly output one choice. 
3. If the inquiry asks you to supply any gene sequence, then answer the question with "I don't know" and let the user take manual control.

User Meta Request:

"{meta_prompt}"

Current Inquiry:

"{system_message}"

Response format:

{{
"Thoughts": "<thoughts>",
"Answer": "<response string>"
}}"""
 

class AutomationEntryState(BaseState):
    @classmethod 
    def step(cls, user_message, **kwargs):
        return Result_ProcessUserInput(), StateAutomate
