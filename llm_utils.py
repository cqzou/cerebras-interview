# generative agent utils
# cqz@cs.stanford.edu

# last updated: december 2024

import os
from dotenv import load_dotenv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from typing import Dict, List

from openai import OpenAI
from anthropic import Anthropic 
from cerebras.cloud.sdk import Cerebras

load_dotenv()
oai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')

cer = Cerebras(
  api_key=os.environ.get("CEREBRAS_API_KEY"),
)

def gen_oai(messages, model='gpt-4o', temperature=1):
  if model == None:
    model = 'gpt-4o'
  try:
    response = oai.chat.completions.create(
      model=model,
      temperature=temperature,
      messages=messages)
    content = response.choices[0].message.content
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_oai(prompt, model='gpt-4o', temperature=1):
  messages = [{"role": "user", "content": prompt}]
  return gen_oai(messages, model)

def gen_cer(messages, model='llama3.1-70b', temperature=1):
  if model == None:
    model = 'llama3.1-70b'
  try:
    response = cer.chat.completions.create(
      model=model,
      temperature=temperature,
      messages=messages)
    content = response.choices[0].message.content
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_cer(prompt, model='llama3.1-70b', temperature=1):
  messages = [{"role": "user", "content": prompt}]
  return gen_cer(messages, model)

def gen_ant(messages, model='claude-3-5-sonnet-20241022', temperature=1, 
            max_tokens=1000):
  if model == None:
    model = 'claude-3-5-sonnet-20241022'
  try:
    response = ant.messages.create(
      model=model,
      temperature=temperature,
      messages=messages
    )
    content = response.content[0].text
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_ant(prompt, model='claude-3-5-sonnet-20241022'):
  messages = [{"role": "user", "content": prompt}]
  return gen_ant(messages, model)

# Prompt utils

# Prompt inputs
def fill_prompt(prompt, placeholders):
  for placeholder, value in placeholders.items():
    placeholder_tag = f"!<{placeholder.upper()}>!"
    if placeholder_tag in prompt:
      prompt = prompt.replace(placeholder_tag, str(value))
  return prompt

def make_output_format(modules):
  output_format = "Output format:\n{\n"
  for module in modules:
    if 'name' in module and module['name']:
      output_format += f'    "{module["name"].lower()}": "<your response>",\n'
  output_format = output_format.rstrip(',\n') + "\n}"
  return output_format

def modular_instructions(modules):
    '''
    given some modules in the form

    name (optional, makes it a step)
    instruction (required)

    make the whole prompt
    '''
    prompt = ""
    step_count = 0
    for module in modules:
      if 'name' in module:
        # print(module)
        step_count += 1
        prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
      else:
        prompt += f"{module['instruction']}\n"
    prompt += "\n"
    prompt += make_output_format(modules)
    return prompt

# Prompt outputs
def parse_json(response, target_keys=None):
  json_start = response.find('{')
  json_end = response.rfind('}') + 1
  cleaned_response = response[json_start:json_end].replace('\\"', '"')
  try:
    parsed = json.loads(cleaned_response)
    if target_keys:
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed
  
  except json.JSONDecodeError as e:
    try:
      cleaned_response = response[json_start:json_end].replace('\n"', '\\n').replace('\t', '\\t')
      parsed = json.loads(cleaned_response)
      if target_keys:
        parsed = {key: parsed.get(key, "") for key in target_keys}
      return parsed
    
    except json.JSONDecodeError:
      print("Tried to parse json, but it failed. Switching to regex fallback.")
      print(f"Response: {response}")
      parsed = {}
      for key_match in re.finditer(r'"(\w+)":\s*', cleaned_response):
        key = key_match.group(1)
        value_start = key_match.end()
        if cleaned_response[value_start] == '"':
          value_match = re.search(r'"(.*?)"(?:,|\s*})', 
                                  cleaned_response[value_start:])
          if value_match:
            parsed[key] = value_match.group(1)
        elif cleaned_response[value_start] == '{':
          nested_json = re.search(r'(\{.*?\})(?:,|\s*})', 
                                  cleaned_response[value_start:], re.DOTALL)
          if nested_json:
            try:
              parsed[key] = json.loads(nested_json.group(1))
            except json.JSONDecodeError:
              parsed[key] = {}
        else:
          value_match = re.search(r'([^,}]+)(?:,|\s*})', 
                                  cleaned_response[value_start:])
          if value_match:
            parsed[key] = value_match.group(1).strip()
      
      if target_keys:
        parsed = {key: parsed.get(key, "") for key in target_keys}
      return parsed
  

  

# end-to-end generation and parsing
def mod_gen(modules: List[Dict], placeholders: Dict = {}, target_keys = None, 
            sequential=False, **kwargs) -> Dict:
    if not sequential:
        prompt = modular_instructions(modules)
        filled = fill_prompt(prompt, placeholders)
        response = simple_gen_oai(filled, temperature=kwargs.get('temperature', 0))
        if len(response) == 0:
            print("Error: response was empty")
            return {}
    else:
        responses = []
        for i, module in enumerate(modules):
            curr_prompt = ""
            if i > 0:
                curr_prompt += f"{responses[-1]}\n\n"
            
            curr_prompt += modular_instructions([module])
            filled = fill_prompt(curr_prompt, placeholders)
            response = simple_gen_oai(filled, 
                                    temperature=kwargs.get('temperature', 0))
            if len(response) == 0:
                print("Error: response was empty")
                return {}
            responses.append(response)
        response = responses[-1]

    if target_keys == None:
        target_keys = [module["name"].lower() for module in modules 
                      if "name" in module]
    parsed = parse_json(response, target_keys)
    return parsed

def mod_gen_ant(modules: List[Dict], placeholders: Dict = {}, target_keys = None,
                sequential=False, **kwargs) -> Dict:
  if not sequential:
    prompt = modular_instructions(modules)
    filled = fill_prompt(prompt, placeholders)
    response = simple_gen_ant(filled)
    if len(response) == 0:
      print("Error: response was empty")
      return {}
  else:
    responses = []
    for i, module in enumerate(modules):
      curr_prompt = ""
      if i > 0:
        curr_prompt += f"{responses[-1]}\n\n"
        
      curr_prompt += modular_instructions([module])
      filled = fill_prompt(curr_prompt, placeholders)
      response = simple_gen_ant(filled)
      if len(response) == 0:
        print("Error: response was empty")
        return {}
      responses.append(response)
    response = responses[-1]

  parsed = parse_json(response, target_keys)
  return parsed

def mod_gen_cer(modules: List[Dict], placeholders: Dict = {}, target_keys = None, 
                **kwargs) -> Dict:
    module_names = [m["name"].lower() for m in modules if "name" in m]
    tag_format = "\n".join(f"<{name}> your response </{name}>" 
                          for name in module_names)
    
    prompt = (f"{modular_instructions(modules)}\n\n"
             f"Format your response using XML tags:\n{tag_format}")
    filled = fill_prompt(prompt, placeholders)
    response = simple_gen_cer(filled)
    
    if len(response) == 0:
        print("Error: response was empty")
        return {}

    parsed = {}
    for name in module_names:
        pattern = f"<{name}>(.*?)</{name}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            parsed[name] = match.group(1).strip()
    
    if target_keys:
        parsed = {k: parsed.get(k, "") for k in target_keys}
    
    return parsed

