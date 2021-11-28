
Clair   = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
Chiffre = ["W","B","H","A","Y","P","O","D","Q","Z","X","N","T","S","F","L","R","U","V","M","C","E","K","J","G","I"]

messageChiffre = "NYVYNYEYVAYNWSSQFSVFSMNYVTYQNNYCUV"
messageDechiffre = ""

for l in messageChiffre:
    index = Chiffre.index(l)
    messageDechiffre = messageDechiffre + Clair[index]

print(messageDechiffre)