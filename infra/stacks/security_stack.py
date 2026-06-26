from aws_cdk import (
    aws_iam as iam,
    aws_bedrock as bedrock,
)
from constructs import Construct


class SecurityConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str):
        super().__init__(scope, construct_id)

        self.lambda_role = iam.Role(
            self,
            "RagLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "textract:*",
                    "bedrock:InvokeModel",
                    "bedrock:ApplyGuardrail",
                    "es:*",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "lambda:InvokeFunction",
                    "logs:*",
                ],
                resources=["*"],
            )
        )

        self.guardrail = bedrock.CfnGuardrail(
            self,
            "RagBedrockGuardrail",
            name="rag-document-guardrail-cdk",
            blocked_input_messaging="I can't help with that request because it violates the safety policy.",
            blocked_outputs_messaging="I can't provide that response because it violates the safety policy.",
            content_policy_config={
                "filtersConfig": [
                    {
                        "type": "HATE",
                        "inputStrength": "MEDIUM",
                        "outputStrength": "MEDIUM",
                    },
                    {
                        "type": "INSULTS",
                        "inputStrength": "MEDIUM",
                        "outputStrength": "MEDIUM",
                    },
                    {
                        "type": "SEXUAL",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "VIOLENCE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "MISCONDUCT",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "PROMPT_ATTACK",
                        "inputStrength": "HIGH",
                        "outputStrength": "NONE",
                    },
                ]
            },
        )

        self.guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "RagBedrockGuardrailVersion",
            guardrail_identifier=self.guardrail.attr_guardrail_id,
            description="Initial production guardrail version",
        )

        self.guardrail_id = self.guardrail.attr_guardrail_id
        self.guardrail_version_number = self.guardrail_version.attr_version
