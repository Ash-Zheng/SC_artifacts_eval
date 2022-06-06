import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple

from torch.utils.cpp_extension import load

# Eff_TT_embedding_cuda = load(name="efficient_tt_table", sources=[
#     "/home/zheng_wang/workspace/PipeDLRM/Eff_TT/Efficient_TT/efficient_kernel_wrap.cpp", 
#     "/home/zheng_wang/workspace/PipeDLRM/Eff_TT/Efficient_TT/efficient_tt_cuda.cu", 
#     ], verbose=True)

Eff_TT_embedding_cuda = load(name="efficient_tt_table", sources=[
    "/workspace/SC_artifacts_eval/models/Efficient_TT/efficient_kernel_wrap.cpp", 
    "/workspace/SC_artifacts_eval/models/Efficient_TT/efficient_tt_cuda.cu", 
    ], verbose=True)

def suggested_tt_shapes(  # noqa C901
    n: int, d: int = 3, allow_round_up: bool = True
) -> List[int]:
    from itertools import cycle, islice

    # pyre-fixme[21]
    from scipy.stats import entropy
    from sympy.ntheory import factorint
    from sympy.utilities.iterables import multiset_partitions

    def _auto_shape(n: int, d: int = 3) -> List[int]:
        def _to_list(x: Dict[int, int]) -> List[int]:
            res = []
            for k, v in x.items():
                res += [k] * v
            return res

        p = _to_list(factorint(n))
        if len(p) < d:
            p = p + [1] * (d - len(p))

        def _roundrobin(*iterables):
            pending = len(iterables)
            nexts = cycle(iter(it).__next__ for it in iterables)
            while pending:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    pending -= 1
                    nexts = cycle(islice(nexts, pending))

        def prepr(x: List[int]) -> Tuple:
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[: N // 2], x[N // 2 :]
            return tuple(_roundrobin(xf, xl))

        raw_factors = multiset_partitions(p, d)
        clean_factors = [prepr(f) for f in raw_factors]
        factors = list(set(clean_factors))
        # pyre-fixme[16]
        weights = [entropy(f) for f in factors]
        i = np.argmax(weights)
        return list(factors[i])

    def _roundup(n: int, k: int) -> int:
        return int(np.ceil(n / 10 ** k)) * 10 ** k

    if allow_round_up:
        weights = []
        for i in range(len(str(n))):
            n_i = _roundup(n, i)
            # pyre-fixme[16]
            weights.append(entropy(_auto_shape(n_i, d=d)))
        i = np.argmax(weights)
        factors = _auto_shape(_roundup(n, i), d=d)
    else:
        factors = _auto_shape(n, d=d)
    return factors


class TT_core_function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        batch_size,
        table_length,
        feature_dim,
        indices,

        tt_p_shapes, 
        tt_q_shapes, 
        tt_ranks, 
        tensor_p_shape, 
        tensor_q_shape, 
        tensor_tt_ranks,
        sorted_idx,
        sorted_key,
        *tt_cores,
    ):
        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks
        ctx.tensor_p_shape = tensor_p_shape
        ctx.tensor_q_shape = tensor_q_shape
        ctx.tensor_tt_ranks = tensor_tt_ranks
        ctx.table_length = table_length
        ctx.feature_dim = feature_dim
        ctx.batch_size = batch_size
        ctx.tt_cores = tt_cores
        ctx.sorted_idx = sorted_idx
        ctx.sorted_key = sorted_key

        ctx.save_for_backward(
            indices,
        )

        output = Eff_TT_embedding_cuda.Eff_TT_forward(
            batch_size,
            table_length, # need
            feature_dim, # need
            indices,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shape,
            tensor_q_shape,
            tensor_tt_ranks,
            list(ctx.tt_cores)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor)-> Tuple[torch.Tensor]:
        
        
        indices = ctx.saved_tensors

        if ctx.sorted_key == None:
            sorted_idx, sorted_key = indices[0].unique(sorted=True, return_inverse=True)
        # Eff_TT_embedding_cuda.Eff_TT_backward(
        # Eff_TT_embedding_cuda.Fused_Eff_TT_backward(
            Eff_TT_embedding_cuda.Fused_Extra_Eff_TT_backward(
                ctx.batch_size,
                ctx.table_length,
                ctx.feature_dim,
                0.1,

                indices[0],
                ctx.tt_p_shapes, 
                ctx.tt_q_shapes,
                ctx.tt_ranks, 
                ctx.tensor_p_shape,
                ctx.tensor_q_shape,
                ctx.tensor_tt_ranks,
                grad_output,
                list(ctx.tt_cores),
                sorted_idx,
                sorted_key,
            )
        else:
            Eff_TT_embedding_cuda.Fused_Extra_Eff_TT_backward(
                ctx.batch_size,
                ctx.table_length,
                ctx.feature_dim,
                0.1,

                indices[0],
                ctx.tt_p_shapes, 
                ctx.tt_q_shapes,
                ctx.tt_ranks, 
                ctx.tensor_p_shape,
                ctx.tensor_q_shape,
                ctx.tensor_tt_ranks,
                grad_output,
                list(ctx.tt_cores),
                ctx.sorted_idx,
                ctx.sorted_key,
            )


        # Eff_TT_embedding_cuda.Eff_TT_backward(
        # # Eff_TT_embedding_cuda.Fused_Eff_TT_backward(
        #     ctx.batch_size,
        #     ctx.table_length,
        #     ctx.feature_dim,
        #     0.1,

        #     indices[0],
        #     ctx.tt_p_shapes, 
        #     ctx.tt_q_shapes,
        #     ctx.tt_ranks, 
        #     ctx.tensor_p_shape,
        #     ctx.tensor_q_shape,
        #     ctx.tensor_tt_ranks,
        #     grad_output,
        #     list(ctx.tt_cores)
        # )

        return tuple(
            [
                None,  # tt_p_shapes
                None,  # tt_q_shapes
                None,  # tt_ranks
                None,  # tensor_p_shape
                None,  # tensor_q_shape
                None,  # tensor_tt_ranks
                None,  # table_length
                None,  # feature_dim
                None,  # batch_size
                None,  # tt_cores
                None,  # learning rate
                None,  # grad_output
                None,  # index
                None,  # keys
                None,  # sorted_index
                None,  # sorted_key
            ]
        )

class Eff_TTEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tt_ranks: List[int],
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        optimizer: str = "SGD",
        learning_rate: float = 0.1,
        weight_dist: str = "uniform",
        device=0,
        batch_size=4096,
    ) -> None:
        super(Eff_TTEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tt_ranks = tt_ranks
        self.num_tt_core = len(self.tt_ranks) + 1
        self.tt_ranks = [1] + tt_ranks + [1]

        self.tt_p_shapes: List[int] = (
            suggested_tt_shapes(num_embeddings, self.num_tt_core)
            if tt_p_shapes is None
            else tt_p_shapes
        )
        self.tt_q_shapes: List[int] = (
            suggested_tt_shapes(embedding_dim, self.num_tt_core,)
            if tt_q_shapes is None
            else tt_q_shapes
        )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_dist = weight_dist
        self.device = device

        Eff_TT_embedding_cuda.init_cuda(device, self.tt_q_shapes, self.tt_ranks, batch_size, embedding_dim)

        # init TT cores 
        self.tt_cores = torch.nn.ParameterList()
        for i in range(self.num_tt_core):
            self.tt_cores.append(
                torch.nn.Parameter(
                    torch.empty(
                        [
                            self.tt_p_shapes[i],
                            self.tt_ranks[i]
                            * self.tt_q_shapes[i]
                            * self.tt_ranks[i + 1],
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
            )
        # print(self.tt_cores[0].shape, self.tt_cores[1].shape, self.tt_cores[2].shape)

        self.reset_parameters()
        self.tensor_p_shape = torch.tensor(self.tt_p_shapes).to(self.device)
        self.tensor_q_shape = torch.tensor(self.tt_q_shapes).to(self.device)
        self.tensor_tt_ranks = torch.tensor(self.tt_ranks).to(self.device)

    def reset_parameters(self):
        if self.weight_dist == "uniform":
            lamb = 2.0 / (self.num_embeddings + self.embedding_dim)
            stddev = np.sqrt(lamb)
            tt_ranks = np.array(self.tt_ranks)
            cr_exponent = -1.0 / (2 * self.num_tt_core)
            var = np.prod(tt_ranks ** cr_exponent)
            core_stddev = stddev ** (1.0 / self.num_tt_core) * var
            for i in range(self.num_tt_core):
                torch.nn.init.uniform_(self.tt_cores[i], 0.0, core_stddev)
    
    def forward(self, indices, offsets=None, unique=None, inverse=None):
        batch_size = indices.shape[0]
        output = TT_core_function.apply(
            batch_size,
            self.num_embeddings,
            self.embedding_dim,
            indices,
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
            self.tensor_p_shape,
            self.tensor_q_shape,
            self.tensor_tt_ranks,
            unique,
            inverse,
            *(self.tt_cores),           
        ).to(self.device)
        return output  
     
