from aws_cdk import (
    Duration,
    aws_lambda as _lambda,
)
from constructs import Construct


class ComputeConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        lambda_role,
        opensearch_host: str,
        guardrail_id: str,
        guardrail_version: str,
    ):
        super().__init__(scope, construct_id)

        dependency_layer = _lambda.LayerVersion(
            self,
            "RagDependencyLayer",
            code=_lambda.Code.from_asset("../layers"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="Dependencies for RAG Lambda functions",
        )

        self.document_processor = _lambda.Function(
            self,
            "DocumentProcessorLambda",
            function_name="rag-document-processor-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../backend/rag-document-processor"),
            role=lambda_role,
            timeout=Duration.seconds(900),
            memory_size=1024,
            layers=[dependency_layer],
            environment={
                "OPENSEARCH_HOST": opensearch_host,
            },
        )

        self.query_api = _lambda.Function(
            self,
            "RagQueryApiLambda",
            function_name="rag-query-api-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../backend/rag-query-api"),
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=512,
            layers=[dependency_layer],
            environment={
                "OPENSEARCH_HOST": opensearch_host,
            },
        )

        self.answer_api = _lambda.Function(
            self,
            "AnswerLambda",
            function_name="rag-answer-api-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../backend/ollamaconnector"),
            role=lambda_role,
            timeout=Duration.seconds(120),
            memory_size=512,
            layers=[dependency_layer],
            environment={
                "GUARDRAIL_ID": guardrail_id,
                "GUARDRAIL_VERSION": guardrail_version,
            },
        )
