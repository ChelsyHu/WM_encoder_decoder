import numpy as np 
import random 

artist_name=['Michelangelo',"Vincent van Gogh","Caravaggio",
        "Rembrandt","Leonardo da Vinci","Johannes Vermeer",
        "Claude Monet","Raphael","Pablo Picasso","Diego Velázquez",
        "Salvador Dalí","Pierre-Auguste Renoir","Peter Paul Rubens",
        "Édouard Manet", "Paul Cézanne", "Edgar Degas",
        "Gustav Klimt", "Edvard Munch", "Henri Matisse","Georges-Pierre Seurat"
        ]
        # 设置词组长度
word_count = 8
random_words = [random.choice(artist_name) for _ in range(word_count)]
print(random_words)