import re

with open("SemEval2018-T4-train-taskA.txt", "r") as f:
    data = f.readlines()

for line in data[1:]:
    words = line.decode('utf-8').split()
    linenumber = int(words[0])
    label = int(words[1])
    sent = words[2:]
    for i in range(len(sent)):
        sent[i] = re.sub(r'^https?:\/\/.*[\r\n]*', 'URL', sent[i])
        sent[i] = re.sub(r'@[a-zA-Z0-9_]+', 'USER', sent[i]);
    if label == 1:
        f = open("irony-corpus-taskA/irony/" + str(linenumber) + ".txt", "w+")
    else:
        f = open("irony-corpus-taskA/non-irony/" + str(linenumber) + ".txt", "w+")
    f.write(' '.join(sent).encode('utf-8'))
    f.close()
    print label