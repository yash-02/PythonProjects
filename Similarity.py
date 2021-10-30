
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import random as rd

max_vocab_length = 1000
max_length = 15

data = ['Hi',
 'Hey',
 'Is anyone there?',
 'Hello',
 'Hay',
 'Bye',
 'See you later',
 'Goodbye',
 'Thanks',
 'Thank you',
 "That's helpful",
 'Thanks for the help',
 'Who are you?',
 'What are you?',
 'Who you are?',
 'what is your name',
 'what should I call you',
 'whats your name?',
 'Could you help me?',
 'give me a hand please',
 'Can you help?',
 'What can you do for me?',
 'I need a support',
 'I need a help',
 'support me please',
 'I need to create a new account',
 'how to open a new account',
 'I want to create an account',
 'can you create an account for me',
 'how to open a new account',
 'have a complaint',
 'I want to raise a complaint',
 'there is a complaint about a service'
 "What's up",
 "What's new"
 'any advise for preparing good resume',
 'how to make impactfull resume',
 'how to crack any interview',
 'what is the keys points to clear imterview',
 "Tell Me About Yourself."
 'How Did You Hear About This Position?',
 'Why Do You Want to Work at This Company?',
 'Why Do You Want This Job?',
 'Why Should We Hire You?',
  'How you doing',
  'How are you',
  'what are you doing'
  "How's everything?",
  "What's going on",
  'How are things going',
  "What's up",
  "What's new"
  'any advise for preparing good resume',
  'how to make impactfull resume',
  'how to crack any interview',
  'what is the keys points to clear imterview',
  "Tell Me About Yourself."
  'How Did You Hear About This Position?',
  'Why Do You Want to Work at This Company?',
  'Why Do You Want This Job?',
  'Why Should We Hire You?',
  'What Can You Bring to the Company?',
  'What Are Your Greatest Strengths?',
'What are the first steps to changing a career?',
'How can I combine my interests or passions at work?',
'How can I become more proactive about my career path?',
'How can I incorporate meaning into my career?',
'What good habits should help me focus on my career?',
'What leaders do you look up to for inspiration?',
'What are some books you can recommend on leadership?',
'How do you keep your team motivated?',
'What qualities are lacking among today’s leaders?',
'How do you continue to grow and develop as a leader?',
'What was your worst leadership decision?',
'What was the worst conflict you had to resolve?',
'What was the biggest leadership risk you took?',
'What was your proudest moment as a leader?',
'What are your current goals as a leader?',
'What do you enjoy most about entrepreneurship? What is hardest about it?',
'What are some mistakes you wish you could have avoided?',
'What advice would you give to newbie entrepreneurs?',
'How do you brainstorm and finalize business ideas?',
'What are the biggest mistakes first-time entrepreneurs can make?',
'How do you plan on growing your business or entrepreneurial mindset?',
'What was the toughest moment in your business journey? How did you overcome it?',
'Is there any popular entrepreneurial advice that you agree/disagree with? Why?',
'Where do you think my strengths lie in?',
'How can I develop the right amount of discipline to achieve my goals in this industry?',
'What are the necessary skills that I should develop to rapidly grow in my career?',
'What are some things in your career that you regret not having done earlier?',
'How do I effectively manage my time and prioritize accordingly?',
]

# d = [ 'any advise for preparing good resume',
#   'how to make impactfull resume',
#   'how to crack any interview',
#   'what is the keys points to clear imterview',
#   "Tell Me About Yourself."
#   'How Did You Hear About This Position?',
#   'Why Do You Want to Work at This Company?',
#   'Why Do You Want This Job?',
#   'Why Should We Hire You?',
#   'What Can You Bring to the Company?',
#   'What Are Your Greatest Strengths?',
#         ]

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                output_mode='int',
                                output_sequence_length=max_length)

text_vectorizer.adapt(data)

tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128,
                             embeddings_initializer="uniform",
                             input_length=max_length,
                             name="embedding_1")

def tensorflow_embedding(sentance):
  vectorize= text_vectorizer([sentance])
  return np.array(embedding(vectorize)).flatten()


def similarity(sentance_1,sentance_2=None,custom = False,dataset=False,display_sentance=False,eul_dis = False):

    """
    It takes a Sentances and return the cosine similarity and also Euclidean distances or not.

    Args:
        sentance_1: first sentance for similarity (must be a string).
        sentance_2: second sentance for similarity (must be a string).
        custom: take a custom sentance or not (default = False).
        dataset: take a dataset for random sentance (default = False).
        display_sentance: return given sentances (defualt = False).
        eul_dis: return Euclidean distances of two sentancesnor not (defualt = False)

    Return:
         A cosine similarity between two sentance and also euclidean distances or not.
         display sentances.

    Example usage:
        Ex1: similarity('What Can You Bring to the Company?',
                    custom=False,
                    dataset=data,
                    display_sentance=True
                    ).
                    
        Ex2: similarity('What Can You Bring to the Company?',
                  'What Are Your Greatest Strengths?',
                  custom=True,
                  eul_dis=True,
                  display_sentance=False)
    """
    if custom == False and dataset == False:
        S1 = rd.choice(data)
        score = cosine_similarity([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0][0]
        eul_d = euclidean_distances([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0][0]

        if eul_dis and display_sentance:
            return score,eul_d,sentance_1,S1

        if display_sentance:
            return score,sentance_1,S1

        if eul_dis :
            return score, eul_d

        else:
            return score

    if dataset:
        S1 = rd.choice(dataset)
        score = cosine_similarity([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0]
        eul_d = euclidean_distances([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0][0]

        if eul_dis and display_sentance:
            return score[0],eul_d,sentance_1,S1

        if display_sentance:
            return score[0],sentance_1,S1
        if eul_dis is True:
            return score[0], eul_d
        else:
            return score

    if custom:
        score = cosine_similarity([tensorflow_embedding(sentance_1)],[tensorflow_embedding(sentance_2)])[0]
        eul_d= euclidean_distances([tensorflow_embedding(sentance_1)],[tensorflow_embedding(sentance_2)])[0][0]

        if eul_dis and display_sentance:
            return score[0],eul_d,sentance_1,sentance_2

        if display_sentance:
            return score[0],sentance_1,sentance_2
        if eul_dis is True:
            return score[0], eul_d
        else:
            return score
        

def similarSenc(sentance):
    """
    It will compare user asked question with data and returns the semantically most similar question.

    Args: user defined question.
    Return: the semantically most similar question.

    """
    result = []
    for i in data:
        score = cosine_similarity([tensorflow_embedding(s)], [tensorflow_embedding(i)])[0]
        result.append(score)

    re = max(result)
    sen = data[result.index(max(result))]
    return re[0],sen

