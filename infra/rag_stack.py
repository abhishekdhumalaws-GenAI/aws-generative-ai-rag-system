from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_apigateway as apigw,
)
from constructs import Construct


class RagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        docs_bucket = s3.Bucket(
            self,
            "RagDocumentsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        chat_table = dynamodb.Table(
            self,
            "ChatHistoryTable",
            table_name="chat-history",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.NUMBER,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        lambda_role = iam.Role(
            self,
            "RagLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonTextractFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonDynamoDBFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonOpenSearchServiceFullAccess"),
            ],
        )
	lambda_role.add_to_policy(
 	    iam.PolicyStatement(
        	actions=["lambda:InvokeFunction"],
       	 	resources=["*"]
	    )
	)
        # OpenSearch domain is not added yet because your existing domain is already working.
        # Next step: we will either import current OpenSearch endpoint or create it in CDK.

        rag_query_lambda = _lambda.Function(
            self,
            "RagQueryApiLambda",
            function_name="rag-query-api-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../lambda/rag-query-api"),
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "OPENSEARCH_HOST": "REPLACE_WITH_OPENSEARCH_ENDPOINT"
            },
        )

        answer_lambda = _lambda.Function(
            self,
            "AnswerLambda",
            function_name="rag-answer-api-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../lambda/ollamaconnector"),
            role=lambda_role,
            timeout=Duration.seconds(120),
            memory_size=512,
        )

#        rag_query_lambda.grant_invoke(answer_lambda)

        api = apigw.RestApi(
            self,
            "RagApi",
            rest_api_name="rag-bedrock-api-cdk",
            deploy_options=apigw.StageOptions(stage_name="prod"),
        )

        query_resource = api.root.add_resource("query")
        query_resource.add_method(
            "POST",
            apigw.LambdaIntegration(answer_lambda),
        )

        CfnOutput(self, "DocumentsBucketName", value=docs_bucket.bucket_name)
        CfnOutput(self, "ApiEndpoint", value=api.url)
