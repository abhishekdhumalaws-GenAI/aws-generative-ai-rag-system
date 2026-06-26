import json
import boto3
import urllib3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

http = urllib3.PoolManager()


def send_response(event, context, status, data=None):
    response_body = {
        "Status": status,
        "Reason": f"See CloudWatch logs: {context.log_stream_name}",
        "PhysicalResourceId": event.get("PhysicalResourceId", context.log_stream_name),
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "Data": data or {},
    }

    encoded_body = json.dumps(response_body).encode("utf-8")

    http.request(
        "PUT",
        event["ResponseURL"],
        body=encoded_body,
        headers={
            "content-type": "",
            "content-length": str(len(encoded_body)),
        },
    )


def lambda_handler(event, context):
    try:
        print("Event:", json.dumps(event))

        request_type = event["RequestType"]

        props = event["ResourceProperties"]
        host = props["OpenSearchHost"]
        index_name = props.get("IndexName", "rag-vector-index")

        region = "us-east-1"

        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",
            session_token=credentials.token,
        )

        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )

        if request_type == "Delete":
            print("Delete request received. Leaving index untouched.")
            send_response(event, context, "SUCCESS", {"Message": "Delete ignored"})
            return

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
                            "engine": "nmslib",
                        },
                    },
                    "bucket": {"type": "keyword"},
                    "key": {"type": "keyword"},
                }
            },
        }

        if client.indices.exists(index=index_name):
            print(f"Index already exists: {index_name}")
        else:
            client.indices.create(index=index_name, body=mapping)
            print(f"Index created: {index_name}")

        send_response(event, context, "SUCCESS", {"IndexName": index_name})

    except Exception as e:
        print("Error:", str(e))
        send_response(event, context, "FAILED", {"Error": str(e)})
