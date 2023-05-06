import sys
import mwxml
import glob

if len(sys.argv) != 3:
	print('La commande prend deux arguments : le chemin du fichier xml à lire, et le chemin du corpus txt à produire')

infileName = sys.argv[1]
outfileName = sys.argv[2]

path = glob.glob(infileName)
def process_dump(dump, path):
    for page in dump:
        for revision in page:
            #yield page.title, revision.id, revision.timestamp, len(revision.text)
            yield page.title, revision.text

with open(outfileName, 'w') as f:
    for page_title, rev_text in mwxml.map(process_dump, path):#paths
        print("\n".join(str(v) for v in [page_title, rev_text]), file=f)
        print("$$\n", file=f) # A modifier ou supprimer si vous souhaitez signaler autrement (ou ne pas signaler) la fin d'un article
