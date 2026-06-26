import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import os

region = "us-east-1"
host = os.environ["OPENSEARCH_HOST"]
index_name = "rag-vector-index"

credentials = boto3.Session().get_credentials()
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

mapping = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            },
            "bucket": {"type": "keyword"},
            "key": {"type": "keyword"}
        }
    }
}

if client.indices.exists(index=index_name):
    print("Deleting existing wrong index...")
    client.indices.delete(index=index_name)

client.indices.create(index=index_name, body=mapping)
print("Created correct index:", index_name)
