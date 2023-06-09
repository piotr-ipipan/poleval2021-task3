def substitution(text:str):
    # replacers = {'-\n':'','\n':' '} #to jest słownik, po przecinku można dodawać kolejne elementy
    # for word, repl in replacers.items():
    #     text = text.replace(word, repl)
    text = text.replace('-\n','')
    text = text.replace('\n',' ')
    return text