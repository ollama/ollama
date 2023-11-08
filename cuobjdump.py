
fo= open(f"names/start.txt","w") 
with open("cuobjdump.org") as fi:
    for l in fi:
        l = l.replace("(","")
        if l.startswith(".entry"):
            p = l.split()
            name = p[1]
            fo= open(f"names/{name}.txt","w" )

        fo.write(l)
            
