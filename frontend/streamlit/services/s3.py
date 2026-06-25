import boto3
from services.config import S3_BUCKET

s3 = boto3.client("s3")

def upload_document(file):
    s3.upload_fileobj(file, S3_BUCKET, file.name)
    return True

