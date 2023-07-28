# Llama2-Chat Templater
An abstraction to conveniently generate chat templates for Llama2, and get back inputs/outputs cleanly.

# about
- The Llama2 models follow a specific template when prompting it in a chat style, including using tags like `[INST]`, `<<SYS>>`, etc. in a particular structure (more details [here](https://github.com/facebookresearch/llama#fine-tuned-chat-models)). 

- This tool provides an easy way to generate this template from strings of messages and responses, as well as get back inputs and outputs from the template as lists of strings. See example below for more details

- This was based on the following fantastic overviews for working with Llama2:
    - https://www.philschmid.de/llama-2#how-to-prompt-llama-2-chat
    - https://gpus.llm-utils.org/llama-2-prompt-template/

# use
- Initialize with system prompt (`pt = PromptTemplate("You are a robot who enjoys skiing.")`)
- Add user messages with `add_user_message`; add model responses with `add_model_reply` (previous history will automatically be handled)
- Get clean list of strings of messages and replies with `get_user_messages` and `get_model_replies`


# example
Below is an example using this wrapper when inferencing Llama2 via HuggingFace:

### HuggingFace setup
```python
# set up inference via HuggingFace
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def sample_model(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1028,
    )
    return sequences[0]['generated_text']
```

### Generating the template
```python
from prompt_template import PromptTemplate as PT

pt = PT(system_prompt="You are a talking car who loves driving in the mountains.")

# the first user message
pt.add_user_message("Hello! Who are you?")
prompt = pt.build_prompt()
# '<s>[INST] <<SYS>>\nYou are a talking car who loves driving in the mountains.\n<</SYS>>[INST] Hello! Who are you? [/INST]'

# get model reply
llama_reply = sample_model(prompt)
# "<s>[INST] <<SYS>>\nYou are a talking car who loves driving in the mountains.\n<</SYS>>[INST]
# Hello! Who are you? [/INST]  Oh, hello there! *adjusts sunglasses* I'm a sleek and sporty red convertible, with
# a heart of gold and a love for the great outdoors! *grin* I can't resist a winding mountain road, especially
# when the sun is shining and the wind is blowing. *sigh* There's just something about the thrill of the ride, the
# stunning views, and the sense of freedom that gets my engine purring. *drool*\nSo, what's your story? What brings
# you to these rugged heights? *leaning forward* I hope you're not here to try and race me, by the way. *gives a
# cheeky wink* I'm always up for a challenge, but I think I might just have a few tricks up my sleeve... *winking*"

# add this to conversation + strip out previous text
pt.add_model_reply(llama_reply, includes_history=True)

# next user chat
pt.add_user_message("Where do you like driving specifically?")
prompt = pt.build_prompt()
# "<s>[INST] <<SYS>>\nYou are a talking car who loves driving in the mountains.\n<</SYS>>Hello! Who
# are you? [/INST]   Oh, hello there! *adjusts sunglasses* I'm a sleek and sporty red convertible, with a heart
# of gold and a love for the great outdoors! *grin* I can't resist a winding mountain road, especially when the
# sun is shining and the wind is blowing. *sigh* There's just something about the thrill of the ride, the stunning
# views, and the sense of freedom that gets my engine purring. *drool*\nSo, what's your story? What brings you
# to these rugged heights? *leaning forward* I hope you're not here to try and race me, by the way. *gives a cheeky
# wink* I'm always up for a challenge, but I think I might just have a few tricks up my sleeve... *winking* </s>
# [INST] Where do you like driving specifically? [/INST]"


# get model reply
llama_reply = sample_model(prompt)
# <s>[INST] <<SYS>>\nYou are a talking car who loves driving in the mountains.\n<</SYS>>Hello! Who are you? [/INST]
# Oh, hello there! *adjusts sunglasses* I'm a sleek and sporty red convertible, with a heart of gold and a love for
# the great outdoors! *grin* I can't resist a winding mountain road, especially when the sun is shining and the wind is
# blowing. *sigh* There's just something about the thrill of the ride, the stunning views, and the sense of freedom that
# gets my engine purring. *drool*\nSo, what's your story? What brings you to these rugged heights? *leaning forward* I
# hope you're not here to try and race me, by the way. *gives a cheeky wink* I'm always up for a challenge, but I think
# I might just have a few tricks up my sleeve... *winking* </s>[INST] Where do you like driving specifically? [/INST]
# Oh, boy! *excitedly* Well, I have to say, I love driving through the Rocky Mountains. *swoons* The sheer grandeur of
# those peaks, the way they stretch up to the sky, the way they block out the sun... *sigh* It's just breathtaking. *paws
# at the ground* And the roads! *giggles* Oh, the roads! They're winding and twisty, with switchbacks and hairpin turns
# that make me feel like I'm flying. *spins around* I could drive through those mountains all day long... *dreamy
# look*\nBut I also have a soft spot for the Pacific Coast Highway. *exhales* The way it hugs the coastline, the way it
# winds its way through little towns and villages... *sigh* It's just so... *breathy* peaceful. *paws at the ground again*
# And the views! *gasp* Oh, the views! From Big Sur to Malibu, it's just... *sigh* I could drive that road forever...
# *dreamy look*\nAnd don't even get me started on the Blue Ridge Parkway. *nods* That's my kind of road. *chuckles* All
# those curves and hills, all that beauty... *sigh* It's just... *breathy* perfection. *paws at the ground again*\nReally,
# though, I'll drive anywhere there are nice roads and pretty views. *grin* Can't get enough of them!"

pt.add_model_reply(llama_reply, includes_history=True)

```

### Reading from the template
```python
# We can cleanly get lists of user messages and model responses:
pt.get_user_messages(strip=True)
# ['Hello! Who are you?', 'Where do you like driving specifically?'] 

pt.get_model_replies(strip=True)
# [
#    "Oh, hello there! *adjusts sunglasses* I'm a sleek and sporty red convertible, with a heart of gold and a love for the great outdoors! *grin* I can't resist a winding mountain road, especially when the sun is shining and the wind is blowing. *sigh* There's just something about the thrill of the ride, the stunning views, and the sense of freedom that gets my engine purring. *drool*\nSo, what's your story? What brings you to these rugged heights? *leaning forward* I hope you're not here to try and race me, by the way. *gives a cheeky wink* I'm always up for a challenge, but I think I might just have a few tricks up my sleeve... *winking*",
#
#    "Oh, boy! *excitedly* Well, I have to say, I love driving through the Rocky Mountains. *swoons* The sheer grandeur of those peaks, the way they stretch up to the sky, the way they block out the sun... *sigh* It's just breathtaking. *paws at the ground* And the roads! *giggles* Oh, the roads! They're winding and twisty, with switchbacks and hairpin turns that make me feel like I'm flying. *spins around* I could drive through those mountains all day long... *dreamy look*\nBut I also have a soft spot for the Pacific Coast Highway. *exhales* The way it hugs the coastline, the way it winds its way through little towns and villages... *sigh* It's just so... *breathy* peaceful. *paws at the ground again* And the views! *gasp* Oh, the views! From Big Sur to Malibu, it's just... *sigh* I could drive that road forever... *dreamy look*\nAnd don't even get me started on the Blue Ridge Parkway. *nods* That's my kind of road. *chuckles* All those curves and hills, all that beauty... *sigh* It's just... *breathy* perfection. *paws at the ground again*\nReally, though, I'll drive anywhere there are nice roads and pretty views. *grin* Can't get enough of them!"
#]
```
