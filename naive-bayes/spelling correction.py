
import re, collections


def text(words):
    return re.findall('[a-z]+', words.lower())

def collection(words):
    model = collections.defaultdict( lambda :1 )
    for word in words:
        model[word] += 1

    return model

# to calculate the word frequencies
Ncollection = collection( text( open('big.txt', encoding='utf-8').read() ) )

# to create words
alpha = 'abcdefghijklmnopqrstuvwxyz'


# to generate the words the user really wants to input when the distance is 1
#   input  n chars
#   generate  n-1 ~ n+1 chars
def edits1(word):
    n = len(word)

    temp1 = [ word[0:i] + word[i+1:] for i in range(n) ]
    # Scenario 1: input an excessive char

    temp2 = [ word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1) ]
    # Scenario 2: the order is wrong during two consecutive chars

    temp3 = [ word[0:i] + c + word[i + 1:] for i in range(n) for c in alpha ]
    # Scenario 3: mistype a char

    temp4 = [ word[0:i] + c + word[i:] for i in range(n+1) for c in alpha ]
    # Scenario 4: miss a char

    temp = temp1 + temp2 + temp3 + temp4
    tempSet = set(temp)
    return tempSet


# to generate the words the user really wants to input when the distance is 2
#   input  n chars
#   generate  n-2 ~ n+2 chars
def edits2(word):
    return [  e2      for e1 in edits1(word)      for e2 in edits1(e1)  ]

# For words, only return the words that are in the collection
def known(words):
    return [word for word in words if word in Ncollection]

# 6. 根据结果判断词频最大的
def correct(word):

    temp1 = edits1(word)
    temp2 = edits2(word)

    t1 = known([word])
    t2 = known(temp1)
    t3 = known(temp2)
    t4 = [word]
    known_words = t1 or t2 or t3 or t4  # return t1 when t1 is not null
                                        # return t2 when t1 is null and t2 is not null
                                        # return t3 when t1 and t2 are null and t3 is not null
                                        # return t4 when t1,t2 and t3 are null and t4 is not null

    print(known_words)
    return max( known_words, key=lambda x : Ncollection[x] )


while True:
    x = input("please input a word: ")  # tha, facd, knoq
    if x == '':
        break
    temp = correct(x)
    print( "The most likely word: ", temp )
