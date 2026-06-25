#!/bin/bash
set -e

FUNC_DIR=$1
ZIP_NAME=$2

rm -rf package
mkdir package

pip install -q -t package opensearch-py requests-aws4auth requests boto3

cp backend/$FUNC_DIR/lambda_function.py package/

cd package
zip -qr ../build/$ZIP_NAME .
cd ..

rm -rf package
echo "Created build/$ZIP_NAME"
