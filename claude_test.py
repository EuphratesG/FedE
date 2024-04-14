import anthropic
import os
import time


start_time = time.time()
#print(os.environ.get("ANTHROPIC_API_KEY"))
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=4096,
    temperature=0.8,
    system="You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
    messages=[
        {"role": "user", "content": "I want you to give me a numerical value expressing the likelihood that subject entity A and object entity B are linked by a predicate relation to form a plausible triple in a knowledge graph.\n \
                I will describe a triplet in the following format: Subject Entity A: xxx, Description: xxx;\n Object Entity B: xxx, Description: xxx;\n Predicate Relation: xxx.\n xxx will be the corresponding texts. \
                Now the triplet is as follows :\n \
                \"Subject Entity A: Ray Liotta, Description: Raymond Allen \"Ray\" Liotta is an American actor, best known for his portrayal of Henry Hill in the crime-drama Goodfellas and for his role as Shoeless Joe Jackson in Field of Dreams. He has won an Emmy Award and been nominated for Golden Globe and Screen Actors Guild awards. He also serves as a producer in some films.;\n Object Entity B: Film Producer, Description: Film producers prepare and then supervise the making of a film before presenting the product to a financing entity or a film distributor. They might be employed by a production company or be independent, yet either way they help the creative people as well as the accounting personnel.\n Predicate Relation: person profession. \"  You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity, like: \n \
                0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
                0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
                0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
                0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
                0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
                Emphasizing, the factor you refer to must be the relation I provided, and the relation between the two entities is unidirectional, with entity A pointing to entity B. If in your view, there is almost no relation between Entity A and Entity B, you may give a low score near 0. If you believe there is a strong logical relation from Entity A to Entity B, you may give a high score near 1. \
                Present your answer using this format: \"value of a plausible triple's likelihood = 0.XXXX\" where 0.XXXX is a value between 0 and 1. \
                Ensure that your answer follows the same format as the example above, and there are significant figures in all decimal places."}
    ]
)
print(message.content[0].text)
end_time = time.time()
iteration_time = end_time - start_time
print("循环执行时间:", iteration_time, "秒")