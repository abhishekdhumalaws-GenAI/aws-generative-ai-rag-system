import json
import boto3
import urllib.parse
import time
import uuid
import re
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

region = "us-east-1"

s3 = boto3.client("s3")
textract = boto3.client("textract")
bedrock = boto3.client("bedrock-runtime")

host = "search-ollama-pamoq2lwd52r3s5bn5jlxczgr4.us-east-1.es.amazonaws.com"
index_name = "rag-vector-index"

credentials = boto3.Session().get_credentials()

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True
)

# -----------------------------
# Extract text from PDF using Textract
# -----------------------------
def extract_text_from_pdf(bucket, key):
    print("Starting Textract job for:", key)

    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = response["JobId"]
    print("Textract JobId:", job_id)

    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        status = result["JobStatus"]
        if status in ["SUCCEEDED", "FAILED"]:
            break
        print("Waiting for Textract job...")
        time.sleep(2)

    if status == "FAILED":
        raise Exception("Textract job failed")

    text = ""
    for block in result["Blocks"]:
        if block["BlockType"] == "LINE":
            text += block["Text"] + "\n"

    print("Extracted text length:", len(text))
    return text

# -----------------------------
# Generate embedding using Bedrock
# -----------------------------
def generate_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(modelId="amazon.titan-embed-text-v1", body=body)
    result = json.loads(response["body"].read())
    return result["embedding"]

# -----------------------------
# Chunk text
# -----------------------------
def chunk_text(text, max_chunk_size=800):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    print("Semantic chunks created:", len(chunks))
    return chunks

# -----------------------------
# Lambda Handler
# -----------------------------
def lambda_handler(event, context):
    # Support both S3 trigger and Step Functions input
    if "Records" in event:
        # S3 event
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    else:
        # Step Functions input
        bucket = event.get("bucket")
        key = event.get("key")
        if not bucket or not key:
            raise ValueError("Step Functions input must include 'bucket' and 'key'")

    print("Processing file:", key)

    # Extract text from PDF
    text = extract_text_from_pdf(bucket, key)

    if not text.strip():
        print("No text extracted from document")
        return {"statusCode": 200, "body": json.dumps("No text extracted")}

    # Chunk text
    chunks = chunk_text(text)

    # Process chunks
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue
        embedding = generate_embedding(chunk)
        doc = {
            "content": chunk,
            "embedding": embedding,
            "source": key,
            "chunk_id": str(uuid.uuid4())
        }
        client.index(index=index_name, id=str(uuid.uuid4()), body=doc)

    print("Document successfully stored in OpenSearch")

    return {"statusCode": 200, "body": json.dumps("Document processed and stored in OpenSearch")}
