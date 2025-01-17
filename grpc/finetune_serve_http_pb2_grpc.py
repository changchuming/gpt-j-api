# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import finetune_serve_http_pb2 as finetune__serve__http__pb2


class FinetuneServeStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Prompt = channel.unary_unary(
                '/endpoints.finetune.serve.FinetuneServe/Prompt',
                request_serializer=finetune__serve__http__pb2.PromptRequest.SerializeToString,
                response_deserializer=finetune__serve__http__pb2.PromptResponse.FromString,
                )


class FinetuneServeServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Prompt(self, request, context):
        """Prompts model.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FinetuneServeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Prompt': grpc.unary_unary_rpc_method_handler(
                    servicer.Prompt,
                    request_deserializer=finetune__serve__http__pb2.PromptRequest.FromString,
                    response_serializer=finetune__serve__http__pb2.PromptResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'endpoints.finetune.serve.FinetuneServe', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FinetuneServe(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Prompt(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/endpoints.finetune.serve.FinetuneServe/Prompt',
            finetune__serve__http__pb2.PromptRequest.SerializeToString,
            finetune__serve__http__pb2.PromptResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
