# corpus split
corpus = "insects" # or: insects

with open("corpus_xml/insects.xml", "r") as f:
    data = f.read()

articles = data.split("<page>")

for i in range(len(articles)):
    with open(f'''corpus/{corpus}_{i}.txt''', "w") as f:
        f.write(articles[i])
