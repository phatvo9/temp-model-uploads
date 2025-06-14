import os
import sys

sys.path.append(os.path.dirname(__file__))
from openai_server_starter import OpenAI_APIServer
from openai_client_wrapper import OpenAIWrapper, build_messages
##################

from typing import Dict, Iterator, List, Union

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_types import (Image, Video, Audio)
from clarifai.runners.utils.data_utils import Param

from openai import OpenAI
from openai.types.chat import (ChatCompletionChunk, ChatCompletion)
from openai.resources.completions import Stream as OpenAIStream


class MyRunner(ModelClass):
  """
  A custom runner that integrates with the Clarifai platform and uses Server inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the  server."""
    os.path.join(os.path.dirname(__file__))
    # Use downloaded checkpoints.
    # Or if you intend to download checkpoint at runtime, set hf id instead. For example:
    # checkpoints = "Qwen/Qwen2-7B-Instruct"
    
    # server args were generated by `upload` module
    server_args = {'modalities': ['text'], 'backend': 'turbomind', 'cache_max_entry_count': 0.9, 'tensor_parallel_size': 1, 
                   'max_prefill_token_num': 4096, 'dtype': 'auto', 'quantization_format': None, 'quant_policy': 0, 
                   'chat_template': None, 'max_batch_size': 16, 'device': 'cuda', 'server_name': '0.0.0.0', 'server_port': 23334, 'checkpoints': 'unsloth/Llama-3.2-1B-Instruct'}
    # if checkpoints == "checkpoints" => assign to checkpoints var aka local checkpoints path
    stage = server_args.get("checkpoints")
    if stage in ["build", "runtime"]:
      #checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
      config_path = os.path.dirname(os.path.dirname(__file__))
      builder = ModelBuilder(config_path, download_validation_only=True)
      checkpoints = builder.download_checkpoints(stage=stage)
      server_args.update({"checkpoints": checkpoints})
      
    if server_args.get("additional_list_args") == ['']:
      server_args.pop("additional_list_args")
    
    modalities = server_args.pop("modalities", ["image", "audio", "video"])
    if not modalities:
      modalities = ["image", "audio", "video"]
    print("Model: Accepted modalities: ", modalities)
    # Start server
    # This line were generated by `upload` module
    self.server = OpenAI_APIServer.from_lmdeploy_backend(**server_args)

    self.client = OpenAIWrapper(
        client=OpenAI(
            api_key="notset",
            base_url=OpenAIWrapper.make_api_url(self.server.host, self.server.port)),
        modalities=modalities)
  
  def set_usage(self, resp):
     if resp.usage and resp.usage.prompt_tokens and resp.usage.completion_tokens:
       self.set_output_context(
         prompt_tokens=resp.usage.prompt_tokens,
         completion_tokens=resp.usage.completion_tokens
       )
  
  
  @ModelClass.method
  def predict(
    self,
    prompt: str = "",
    images: List[Image] = [],
    audios: List[Audio] = [],
    videos: List[Video] = [],
    chat_history: List[Dict] = [],
    audio: Audio = None,
    video: Video = None,
    image: Image = None,
    system_prompt: str = Param(default="", description="The system-level prompt used to define the assistant's behavior."),
    max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
    temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response.", ),
    top_p: float = Param(default=0.9, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", ),
  )-> str:
    """Method to call from UI
    """

    completion = self.client.cl_custom_chat(
      prompt=prompt,
      system_prompt=system_prompt,
      images=[image] if image else images,
      audios=[audio] if audio else audios,
      videos=[video] if video else videos,
      chat_history=chat_history,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      stream=False
    )
    
    self.set_usage(completion)
    
    return completion.choices[0].message.content
  

  @ModelClass.method
  def generate(
    self,
    prompt: str = "",
    images: List[Image] = [],
    audios: List[Audio] = [],
    videos: List[Video] = [],
    chat_history: List[Dict] = [],
    audio: Audio = None,
    video: Video = None,
    image: Image = None,
    system_prompt: str = Param(default="", description="The system-level prompt used to define the assistant's behavior."),
    max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
    temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response.", ),
    top_p: float = Param(default=0.9, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", ),
  ) -> Iterator[str]:
    """Method to call generate from UI
    """
    stream_completion = self.client.cl_custom_chat(
        prompt=prompt,
        system_prompt=system_prompt,
        images=[image] if image else images,
        audios=[audio] if audio else audios,
        videos=[video] if video else videos,
        chat_history=chat_history,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )

    for chunk in stream_completion:
      if chunk.choices and chunk.choices[0].delta.content is not None:
        self.set_usage(chunk)
        yield str(chunk.choices[0].delta.content)
    

  @ModelClass.method
  def chat(
      self,
      messages: List[dict],
      frequency_penalty: float = 0.,
      max_tokens: int = None,
      temperature: float = 1.,
      top_p: float = 0.95,
      presence_penalty: float = 0.,
      logprobs: bool = False,
      top_logprobs: int = None,
      max_completion_tokens: int = None,
      modalities: List[str] = ["text"],
      reasoning_effort: str = "none",
      response_format: dict = None,
      tool_choice: str = 'none',
      tools: List[dict] = [],
      extra_body: dict = {},
  ) -> dict:
    """
    """

    completion = self.client.chat(
        messages=messages,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        modalities=modalities,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
        tool_choice=tool_choice,
        tools=tools,
        extra_body=extra_body,
        stream=False
    )
    self.set_usage(completion)
    
    return completion.model_dump()

  @ModelClass.method
  def stream_chat(
      self,
      messages: List[dict],
      temperature: float = 1.,
      top_p: float = 0.95,
      max_tokens: int = None,
      frequency_penalty: float = 0.,
      presence_penalty: float = 0.,
      logprobs: bool = False,
      top_logprobs: int = None,
      max_completion_tokens: int = None,
      modalities: List[str] = ["text"],
      reasoning_effort: str = "medium",
      response_format: dict = None,
      tool_choice: str = 'none',
      tools: List[dict] = [],
      stream_options: dict = {},
      extra_body: dict = {},
  ) -> Iterator[list]:
    """
    """

    iter_responses = self.client.chat(
        messages=messages,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        modalities=modalities,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
        tool_choice=tool_choice,
        tools=tools,
        stream_options=stream_options,
        extra_body=extra_body,
        stream=True
    )

    for each in iter_responses:
      self.set_usage(each)
      yield each.model_dump()

  
  def test(self):
    const_data = dict(
      image=dict(
        images=[
          Image(url="https://samples.clarifai.com/metro-north.jpg")
        ],
        image=Image(url="https://samples.clarifai.com/metro-north.jpg")
      ),
      video=dict(videos=[
          Video(url="https://samples.clarifai.com/beer.mp4")
      ]),
      audio=dict(audios=[
          Audio(
              url="https://samples.clarifai.com/GoodMorning.wav")
      ])
    )
    
    build_modalities_kwargs = {}
    for each in self.client.modalities:
      if each in const_data:
        build_modalities_kwargs.update(const_data.get(each))
    prompt = f"Describe this {', '.join(self.client.modalities)}" \
      if build_modalities_kwargs else "Hi there, how are you?"
    build_modalities_kwargs.update(dict(prompt=prompt))
  
    # dummy history
    chat_history = [
      dict(content="You're an assistant.", role="system"),
      dict(content="Hi.", role="user"),
      dict(content="Hi.", role="assistant"),
    ]
    # Test predict
    logger.info("# 1/4: Test 'predict'.\n")
    print(self.predict(**build_modalities_kwargs, max_tokens=512, temperature=0.9, top_p=0.9, system_prompt="You are an Orange Cat."))
    print("---"*5)
    
    logger.info("# 1.2/4: Test 'predict' History.\n")
    print(self.predict(**build_modalities_kwargs, chat_history=chat_history, max_tokens=512, temperature=0.9, top_p=0.9, system_prompt="You are an Orange Cat."))
    print("---"*5)
    
    # Test generate
    logger.info("# 2/4: Test 'generate'.\n")
    for each in self.generate(**build_modalities_kwargs, max_tokens=512, temperature=0.9, top_p=0.9, system_prompt=""):
      print(each, flush=True, end='')
    print("---"*5)
    
    logger.info("# 2.2/4: Test 'generate' History.\n")
    for each in self.generate(**build_modalities_kwargs, chat_history=chat_history, max_tokens=512, temperature=0.9, top_p=0.9, system_prompt=""):
      print(each, flush=True, end='')
    print("---"*5)
    
    
    # Test chat
    messages = build_messages(**build_modalities_kwargs, max_tokens=512, temperature=0.9, top_p=0.9)
    logger.info("# 3/4: Test 'chat'.\n")
    completions = self.chat(messages=messages)
    print(completions)
    print("---"*5)
    # Test chat_with_stream
    logger.info("# 4/4: Test 'chat_with_stream'.\n")
    for each in self.stream_chat(messages, max_tokens=512, temperature=0.9, top_p=0.9):
      if each["choices"]:
        print(each["choices"][0]["delta"].get("content", ""), flush=True, end='')
    print("---"*5)