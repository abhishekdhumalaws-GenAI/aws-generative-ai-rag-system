import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

region = "us-east-1"
service = "es"  # OpenSearch Service (non-serverless)

bedrock = boto3.client("bedrock-runtime", region_name=region)

host = "search-ollama-pamoq2lwd52r3s5bn5jlxczgr4.us-east-1.es.amazonaws.com"
index = "rag-vector-index"

session = boto3.Session()
credentials = session.get_credentials()

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)


def generate_embedding(text: str):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def lambda_handler(event, context):
    try:
        # Parse incoming event
        body = json.loads(event["body"])
        question = body.get("query", "")

        # Generate embedding
        embedding = generate_embedding(question)

        # KNN search in OpenSearch
        search_body = {
            "size": 3,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": 3
                                }
                            }
                        }
                    ],
                    "should": [
                        {"match": {"content": question}}
                    ]
                }
            }
        }

        response = client.search(index=index, body=search_body)

        # Wrap each result as {"content": "..."} for Lambda3 compatibility
        results = [{"content": hit["_source"]["content"]} for hit in response["hits"]["hits"]]

        # Safe fallback if no results
        if not results:
            results = [{"content": ""}]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results)
        }

    except Exception as e:
        print("Lambda2 error:", e)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
