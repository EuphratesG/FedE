
from openai import OpenAI
import os

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

chat_completion = client.chat.completions.create(
  messages=[
    {
      "role": "system",
      "content": "You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
    },
    {
      "role": "user",
      "content": "I will describe a triplet in the following format: Entity A: xxx, Description: xxx;\n Entity B: xxx, Description: xxx;\n Relation: xxx.\n xxx will be the corresponding texts. \
I want you to give me a numerical value expressing the likelihood that entity A and Entity B establish a directional logical relationship through an intermediary relation to form a plausible triple in a knowledge graph.\n \
The triplet is as follows :\n \
\"Entity A: Ray Liotta, Description: Raymond Allen \"Ray\" Liotta is an American actor, best known for his portrayal of Henry Hill in the crime-drama Goodfellas and for his role as Shoeless Joe Jackson in Field of Dreams. He has won an Emmy Award and been nominated for Golden Globe and Screen Actors Guild awards. He also serves as a producer in some films.;\n Entity B: Film Producer-GB, Description: Film producers prepare and then supervise the making of a film before presenting the product to a financing entity or a film distributor. They might be employed by a production company or be independent, yet either way they help the creative people as well as the accounting personnel.;\n Relation: person profession.\n\"  You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity, like: \n \
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
Present your answer using this format: \"value of a plausible triple's likelihood = 0.XXXX\" where 0.XXXX is a value between 0 and 1. \
Ensure that the numerical answer you provide has the same precision as the example format. Emphasizing, the factor you refer to must be the relation I provided. If in your view, there is almost no relation between Entity A and Entity B as provided, you may give a low score near 0. If you believe there is a strong logical relationship from Entity A to Entity B, you may give a high score near 1. [/INST]"
    }
  ],
  model="google/gemma-7b-it"
)

print(chat_completion.choices[0].message.content)
#   prompt="<s>[INST] <<SYS>>\
# You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.\
# <</SYS>> \
# I will describe a triplet in the following format: Entity A: xxx, Description: xxx;\n Entity B: xxx, Description: xxx;\n Relation: xxx.\n xxx will be the corresponding texts. \
# I want you to give me a numerical value expressing the likelihood that entity A and Entity B establish a directional logical relationship through an intermediary relation to form a plausible triple in a knowledge graph.\n \
# The triplet is as follows :\n \
# \"Entity A: Ray Liotta, Description: Raymond Allen \"Ray\" Liotta is an American actor, best known for his portrayal of Henry Hill in the crime-drama Goodfellas and for his role as Shoeless Joe Jackson in Field of Dreams. He has won an Emmy Award and been nominated for Golden Globe and Screen Actors Guild awards. He also serves as a producer in some films.;\n Entity B: Film Producer-GB, Description: Film producers prepare and then supervise the making of a film before presenting the product to a financing entity or a film distributor. They might be employed by a production company or be independent, yet either way they help the creative people as well as the accounting personnel.;\n Relation: person profession.\n\"  You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity, like: \n \
# 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
# 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
# 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
# 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
# 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
# Present your answer using this format: \"value of a plausible triple's likelihood = 0.XXXX\" where 0.XXXX is a value between 0 and 1. \
# Ensure that the numerical answer you provide has the same precision as the example format. Emphasizing, the factor you refer to must be the relation I provided. If in your view, there is almost no relation between Entity A and Entity B as provided, you may give a low score near 0. If you believe there is a strong logical relationship from Entity A to Entity B, you may give a high score near 1. [/INST]",