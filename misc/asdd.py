import requests as r
from Bio import SeqIO
from io import StringIO

cID = 'P0AAK7'

baseUrl = "http://www.uniprot.org/uniprot/"
currentUrl = baseUrl + cID + ".fasta"
response = r.post(currentUrl)
cData = ''.join(response.text)
print(int(cData.split("PE=")[1].split(" ")[0]))
