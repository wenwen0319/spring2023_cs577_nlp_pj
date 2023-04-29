from stanfordcorenlp import StanfordCoreNLP as scn
nlp = scn(r'/stanford-corenlp-full-2018-10-05/')

sentence = 'His acting was good but script was poor'

print
print 'Part of Speech:'
print nlp.pos_tag(sentence)

print
print 'Dependency Parsing:'
print nlp.dependency_parse(sentence)
