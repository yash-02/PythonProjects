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
        ]

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


sample_sentence = "There's a flood in my street!"


def tensorflow_embedding(sentance):
  vectorize= text_vectorizer([sentance])
  return np.array(embedding(vectorize)).flatten()

sentc = tensorflow_embedding('Why Do You Want This Job')



def similarity(sentance_1,sentance_2=None,custom = False,dataset=False,display_sentance=False):

    """
    It takes a two Sentances and return the cosine similarity of it.

    Args:
        sentance_1: first sentance for similarity (must be a string).
        sentance_2: second sentance for similarity (must be a string).
        custom: take a custom sentance or not (default = False).
        dataset: take a dataset for random sentance (default = False).
        display_sentance: return given sentances (defualt = False).

    Return:
         A cosine similarity between two sentance.
         display sentances.

    Example usage:
        similarity('What Can You Bring to the Company?',
                    custom=False,
                    dataset=data,
                    display_sentance=True
                    )
    """
    if custom == False and dataset == False:
        S1 = rd.choice(data)
        score = cosine_similarity([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0]
        if display_sentance:
            return score[0],sentance_1,S1
        else:
            return score

    if dataset:
        S1 = rd.choice(dataset)
        score = cosine_similarity([tensorflow_embedding(sentance_1)], [tensorflow_embedding(S1)])[0]
        if display_sentance:
            return score[0],sentance_1,S1
        else:
            return score

    if custom:
        score = cosine_similarity([tensorflow_embedding(sentance_1)],[tensorflow_embedding(sentance_2)])[0]
    # eul_dis= euclidean_distances([tensorflow_embedding(w1)],[tensorflow_embedding(w2)])[0][0]
        return score

