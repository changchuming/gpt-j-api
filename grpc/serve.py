import argparse
from concurrent import futures
import os
import time

from google.protobuf import struct_pb2
import grpc

import finetune_serve_pb2
import finetune_serve_pb2_grpc

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_CKPT_PATH = "gs://gpt-j-6b-checkpoints/finetune_end_early_slim/step_501/"
_PARAMS = {
    "layers": 28,
    "d_model": 4096,
    "n_heads": 16,
    "n_vocab": 50400,
    "norm": "layernorm",
    "pe": "rotary",
    "pe_rotary_dims": 64,
    "seq": 2048,
    "cores_per_replica": 8,
    "per_replica_batch": 1,
}


class FinetuneServeServicer(finetune_serve_pb2_grpc.FinetuneServeServicer):
    """Implements the FinetuneServe API server."""
    def __init__(self, network, tokenizer):
        self._network = network
	self._tokenizer = tokenizer

    def Prompt(self, request: finetune_serve_pb2.PromptRequest, context):
        response = finetune_serve_pb2.PromptResponse()

        top_p = request.top_p if request.top_p != 0 else 0.9
        temperature = request.temperature if request.temperature != 0 else 1.0
        token_max_length = request.token_max_length if request.token_max_length != 0 else 512

        start = time.time()
        tokens = self._tokenizer.encode(request.prompt)
        provided_ctx = len(tokens)
        if token_max_length + provided_ctx > 2048:
            return InvalidArgumentError("Length of tokens specified exceeds maximum length.")
        pad_amount = seq - provided_ctx
        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
        batched_tokens = np.array([padded_tokens] * total_batch)
        length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

        output = self._network.generate(
            batched_tokens,
            length,
            request.token_max_length,
            {
                "top_p": np.ones(total_batch) * top_p,
                "temp": np.ones(total_batch) * temperature,
            },
        )

        text = self._tokenizer.decode(output[1][0][0, :, 0])

        # A simple technique to stop at stop_sequence without modifying the underlying model
        if request.stop_sequence != "" and request.stop_sequence in text:
            text = text.split(request.stop_sequence)[0] + request.stop_sequence

        response.model = "GPT-J-6B"
        response.compute_time = time.time() - start
        response.response = text
        response.prompt = context
        response.token_max_length = token_max_length
        response.temperature = temperature
        response.top_p = top_p
        response.stop_sequence = stop_sequence

        print(response)
        return response



def create_network():
    """Creates a transformer network."""
    per_replica_batch = _PARAMS["per_replica_batch"]
    cores_per_replica = _PARAMS["cores_per_replica"]
    seq = _PARAMS["seq"]


    _PARAMS["sampler"] = nucleaus_sample

    # here we "remove" the optimizer parameters from the model (as we don't need them for inference)
    _PARAMS["optimizer"] = optax.scale(0)

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ("dp", "mp")), ())

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica

    network = CausalTransformer(_PARAMS)
    network.state = read_ckpt(network.state, _CKPT_PATH, devices.shape[1])
    del network.state["opt_state"]
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
    return network, tokenizer


def serve(port, shutdown_grace_duration):
    """Configures and runs the FinetuneServe API server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    network, tokenizer = create_network()
    finetune_serve_pb2_grpc.add_FinetuneServeServicer_to_server(
        FinetuneServeServicer(network, tokenizer), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    print('Listening on port {}'.format(port))

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(shutdown_grace_duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--port', type=int, default=None,
        help='The port to listen on.'
             'If arg is not set, will listen on the $PORT env var.'
             'If env var is empty, defaults to 8000.')
    parser.add_argument(
        '--shutdown_grace_duration', type=int, default=5,
        help='The shutdown grace duration, in seconds')

    args = parser.parse_args()

    port = args.port
    if not port:
        port = os.environ.get('PORT')
    if not port:
        port = 8000

    serve(port, args.shutdown_grace_duration)
