#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.rag_platform_stack import RagPlatformStack

app = cdk.App()

RagPlatformStack(
    app,
    "RagDocumentIntelligencePlatformStack",
    env=cdk.Environment(
        account="623593083974",
        region="us-east-1",
    ),
)

app.synth()
