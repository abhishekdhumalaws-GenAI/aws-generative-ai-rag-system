from aws_cdk import (
    CfnOutput,
    aws_apigateway as apigw,
)
from constructs import Construct


class ApiConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, answer_lambda):
        super().__init__(scope, construct_id)

        self.api = apigw.RestApi(
            self,
            "RagApi",
            rest_api_name="rag-bedrock-api-cdk",
            deploy_options=apigw.StageOptions(stage_name="prod"),
        )

        query = self.api.root.add_resource("query")

        query.add_method(
            "POST",
            apigw.LambdaIntegration(answer_lambda),
            api_key_required=True,
        )

        self.api_key = self.api.add_api_key("RagApiKey")

        self.usage_plan = self.api.add_usage_plan(
            "RagUsagePlan",
            name="rag-usage-plan-cdk",
            throttle=apigw.ThrottleSettings(
                rate_limit=2,
                burst_limit=5,
            ),
            quota=apigw.QuotaSettings(
                limit=100,
                period=apigw.Period.DAY,
            ),
        )

        self.usage_plan.add_api_key(self.api_key)

        self.usage_plan.add_api_stage(
            stage=self.api.deployment_stage,
        )

        CfnOutput(self, "ApiEndpoint", value=self.api.url)
