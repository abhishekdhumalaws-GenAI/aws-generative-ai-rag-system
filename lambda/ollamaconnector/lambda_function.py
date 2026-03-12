import json
import requests
import boto3
import time

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("chat-history")

OLLAMA_API_URL = "http://98.93.204.123:11434/api/generate"
MODEL_NAME = "deepseek-r1:8b"

lambda_client = boto3.client("lambda")

TOP_K_CHUNKS = 5


def lambda_handler(event, context):
    try:
        path = event.get("path", "")

        # -----------------------------
        # Open WebUI model list
        # -----------------------------
        if path.endswith("/api/tags"):
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "models": [
                        {
                            "name": MODEL_NAME,
                            "model": MODEL_NAME,
                            "modified_at": "2024-01-01T00:00:00Z",
                            "size": 0
                        }
                    ]
                })
            }

        # -----------------------------
        # Chat request
        # -----------------------------
        if path.endswith("/api/chat") or "body" in event:
            body = json.loads(event.get("body", "{}"))

            # Support old 'query' style for backward compatibility
            if "messages" in body:
                messages = body.get("messages", [])
            elif "query" in body:
                messages = [{"role": "user", "content": body.get("query", "")}]
            else:
                messages = []

            if not messages:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "No messages provided"})
                }

            user_query = messages[-1]["content"].strip()
            if not user_query:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "Query is empty"})
                }

            # -----------------------------
            # Retrieve conversation history safely
            # -----------------------------
            history_text = ""
            try:
                session_id = body.get("chat_id", "default")
                response = table.query(
                    KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
                    ScanIndexForward=True
                )
                items = response.get("Items", [])
                for item in items[-5:]:
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    history_text += f"User: {q}\nAssistant: {a}\n"
            except Exception as e:
                print("History fetch error:", e)

            # -----------------------------
            # Call Lambda 2 for RAG retrieval safely
            # -----------------------------
            try:
                response = lambda_client.invoke(
                    FunctionName="rag-query-api",
                    InvocationType="RequestResponse",
                    Payload=json.dumps({
                        "body": json.dumps({"query": user_query})
                    })
                )
                payload = json.loads(response["Payload"].read())
                body_str = payload.get("body", "[]")
                try:
                    results = json.loads(body_str)
                except Exception as e:
                    print("RAG parse error:", e, body_str)
                    results = []
                chunks = [r.get("content", "") for r in results if isinstance(r, dict)]
            except Exception as e:
                print("RAG Lambda invoke error:", e)
                chunks = []

            if not chunks:
                answer = "No relevant information found."
            else:
                top_chunks = chunks[:TOP_K_CHUNKS]

            context_text = "\n".join(chunks[:TOP_K_CHUNKS]) if chunks else ""

            # -----------------------------
            # Build prompt for LLM
            # -----------------------------
            prompt = f"""
You are a helpful AI assistant.

Use ONLY the context provided to answer the question.

Return a clear answer in plain text.
Do NOT return JSON.
Do NOT include follow_up questions.

Context:
{context_text}

Conversation History:
{history_text}

Question:
{user_query}

Answer:
"""
            print("Prompt sent to Ollama:", prompt)

            # -----------------------------
            # Call Ollama
            # -----------------------------
            ollama_payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
            ollama_response = requests.post(
                OLLAMA_API_URL,
                headers={"Content-Type": "application/json"},
                json=ollama_payload,
                timeout=60
            )

            if ollama_response.status_code != 200:
                return {
                    "statusCode": 500,
                    "body": json.dumps({
                        "error": "Ollama API failed",
                        "details": ollama_response.text
                    })
                }

            ollama_result = ollama_response.json()
            answer = ollama_result.get("response") or ollama_result.get("text") or ""

            # -----------------------------
            # Prevent JSON outputs
            # -----------------------------
            if answer.strip().startswith("{"):
                try:
                    parsed = json.loads(answer)
                    answer = parsed.get("answer") or parsed.get("response") or str(parsed)
                except:
                    pass

            # -----------------------------
            # Save chat history safely
            # -----------------------------
            try:
                session_id = body.get("chat_id", "default")
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

            # -----------------------------
            # Return WebUI-compatible response
            # -----------------------------
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "model": MODEL_NAME,
                    "message": {"role": "assistant", "content": answer},
                    "done": True
                })
            }

        return {
            "statusCode": 404,
            "body": json.dumps({"error": "Route not found"})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
