class Journel:
    def __init__(self):
        self.entries=[]
        self.count=0

    def add_entry(self,text):
        self.count += 1
        self.entries.append(f"{self.count}:{text}")

    def remove_entry(self,pos):
        del self.entries[pos]

    def __str__(self):
        return "\n".join(self.entries)

    def save(self,filename):
        file=open(filename, 'w')
        file.write(str(self))
        file.close()

    def load(self,filename):
        pass

    def low_from_web(self,uri):
        pass


class PersistenceManager:
    @staticmethod
    def save_to_file(journel, filename):
        file=open(filename, 'w')
        file.write(str(journel))
        file.close()


    



J=Journel()
J.add_entry("I eat apple")
J.add_entry("I eat banana")
J.add_entry("I eat cherry")
print(J)

file = r'C:\Users\nekumar2\distributed_state_ledger\journel.txt'

PersistenceManager.save_to_file(J,file)

with open(file) as fh:
    print(fh.read())