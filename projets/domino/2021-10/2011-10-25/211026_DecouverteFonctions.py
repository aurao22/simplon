mots = ['eddard', 'catelyn', 'robb', 'sansa', 'arya', 'brandon',
        'rickon', 'theon', 'rorbert', 'cersei', 'tywin', 'jaime',
        'tyrion', 'shae', 'bronn', 'lancel', 'joffrey', 'sandor',
        'varys', 'renly', 'a']


def displayListeOfTuple(liste):
    for t in liste:
        print(t)



def mots_lettre_position(liste, lettre, position):
    res = []

    for mot in liste:

        pos = position - 1
        if (pos >= len(mot)):
             continue

        if(mot[pos] == lettre):
            res.append(mot)
    return res


print(mots_lettre_position(mots, "y", 2))