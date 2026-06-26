from aws_cdk import (
    Duration,
    aws_cloudwatch as cloudwatch,
)
from constructs import Construct


class MonitoringConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        document_processor,
        query_api,
        answer_api,
        api,
    ):
        super().__init__(scope, construct_id)

        self.dashboard = cloudwatch.Dashboard(
            self,
            "RagDashboard",
            dashboard_name="rag-document-intelligence-dashboard",
        )

        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Invocations",
                left=[
                    document_processor.metric_invocations(),
                    query_api.metric_invocations(),
                    answer_api.metric_invocations(),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Lambda Errors",
                left=[
                    document_processor.metric_errors(),
                    query_api.metric_errors(),
                    answer_api.metric_errors(),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Lambda Duration",
                left=[
                    document_processor.metric_duration(),
                    query_api.metric_duration(),
                    answer_api.metric_duration(),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="API Gateway Count",
                left=[
                    api.metric_count(period=Duration.minutes(5))
                ],
                width=12,
            ),
        )
