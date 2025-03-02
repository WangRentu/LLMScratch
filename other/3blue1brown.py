import gensim.downloader

print(gensim.downloader.BASE_DIR)

model = gensim.downloader.load("glove-wiki-gigaword-50")

print(model["tower"])