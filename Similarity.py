
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

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
 'there is a complaint about a service']

# text_vectorizer = TextVectorization(max_tokens=None,
#                                     standardize="lower_and_strip_punctuation",
#                                     split="whitespace",
#                                     ngrams=None,
#                                     output_mode="int",
#                                     output_sequence_length=None)

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

# print(embedding(text_vectorizer(['hello world, its a very beutiful day, i like it'])))
sample_sentence = "There's a flood in my street!"
# print(text_vectorizer([sample_sentence]))

def tensorflow_embedding(sentance):
  vectorize= text_vectorizer([sentance])
  return np.array(embedding(vectorize)).flatten()

sentc = tensorflow_embedding('Why Do You Want This Job')
# print(sentc.shape)
def similarity(w1,w2):
  # print('called')
    """
    Args:
        sentance 1: fist sentance
        sectance 2: second sentance

    :return:
        Return a cosine similarity between two sentance
    """
    score = cosine_similarity([tensorflow_embedding(w1)],[tensorflow_embedding(w2)])
    # eul_dis= euclidean_distances([tensorflow_embedding(w1)],[tensorflow_embedding(w2)])[0][0]
    return score



# if __name__ == '__main__':