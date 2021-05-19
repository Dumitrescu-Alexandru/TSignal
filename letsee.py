import urllib.parse
import urllib.request

url = 'https://www.uniprot.org/uploadlists/'

# params = {
# 'from': 'ACC+ID',
# 'to': 'ENSEMBL_ID',
# 'format': 'tab',
# 'query': 'P40925 P40926 O43175 Q9UM73 P97793'
# }

params = {
'from': 'VEUPATHDB_ID',
'to': 'ACC+ID',
'format': 'tab',
'query': 'HostDB:ENSG00000006652.13'
}


data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
   response = f.read()
print(response.decode('utf-8'))