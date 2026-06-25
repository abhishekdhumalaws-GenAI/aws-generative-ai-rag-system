#!/usr/bin/env python3
import aws_cdk as cdk
from rag_stack import RagStack

app = cdk.App()

RagStack(
    app,
    "RagBedrockStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region="us-east-1"
    )
)

app.synth()
