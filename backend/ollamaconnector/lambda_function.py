import json
import boto3
import time
from boto3.dynamodb.conditions import Key

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
lambda_client = boto3.client("lambda")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("chat-history")

MODEL_ID = "amazon.nova-lite-v1:0"
TOP_K_CHUNKS = 5


def call_nova(prompt):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 800,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]


def lambda_handler(event, context):
    try:
        path = event.get("path", "")
        body = json.loads(event.get("body", "{}"))

        if path.endswith("/api/tags"):
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "models": [{"name": MODEL_ID, "model": MODEL_ID}]
                })
            }

        if "messages" in body:
            messages = body.get("messages", [])
            user_query = messages[-1]["content"] if messages else ""
        else:
            user_query = body.get("query", "")

        user_query = user_query.strip()

        if not user_query:
            return {"statusCode": 400, "body": json.dumps({"error": "Query is empty"})}

        session_id = body.get("chat_id", "default")

        history_text = ""
        try:
            history = table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=True
            )
            for item in history.get("Items", [])[-5:]:
                history_text += f"User: {item.get('question','')}\nAssistant: {item.get('answer','')}\n"
        except Exception as e:
            print("History error:", e)

        rag_response = lambda_client.invoke(
            FunctionName="rag-query-api",
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "body": json.dumps({"query": user_query})
            })
        )

        payload = json.loads(rag_response["Payload"].read())
        results = json.loads(payload.get("body", "[]"))
        chunks = [r.get("content", "") for r in results if isinstance(r, dict)]
        context_text = "\n\n".join(chunks[:TOP_K_CHUNKS])

        prompt = f"""
You are a helpful RAG assistant.

Use ONLY the provided document context to answer.
If the answer is not present in the context, say: "I could not find this information in the uploaded document."

Document Context:
{context_text}

Conversation History:
{history_text}

Question:
{user_query}

Answer:
"""

        answer = call_nova(prompt)

        try:
            table.put_item(
                Item={
                    "session_id": session_id,
                    "timestamp": int(time.time()),
                    "question": user_query,
                    "answer": answer
                }
            )
        except Exception as e:
            print("DynamoDB save error:", e)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "model": MODEL_ID,
                "message": {"role": "assistant", "content": answer},
                "answer": answer,
                "done": True
            })
        }

    except Exception as e:
        print("Error:", e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
