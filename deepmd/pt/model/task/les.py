# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)

log = logging.getLogger(__name__)


@GeneralFitting.register("les")
@fitting_check_output
class LesFittingNet(GeneralFitting):
    """Fitting network for Latent Electrostatic (LES) with dual-MLP branches.

    This class builds two parallel, architecturally identical MLP branches that
    operate on the same per-atom descriptor input:
    - Energy branch: predicts atomic energy contributions (`energy`).
    - Charge branch: predicts latent atomic charge (`q`).
    Both branches share the same input and output dimensions to ensure consistency.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int, optional
        Output dimension of each branch. Default is 1.
    neuron : list[int], optional
        Hidden layer sizes of both MLP branches. Default is [128, 128, 128].
    bias_atom_e : torch.Tensor, optional
        Average energy per atom for each element for the energy branch.
        Shape should be [ntypes, dim_out] when `mixed_types` is True; type-specific otherwise.
    resnet_dt : bool, optional
        Use time-step in ResNet construction. Default is True.
    numb_fparam : int, optional
        Number of frame parameters. Default is 0.
    numb_aparam : int, optional
        Number of atomic parameters. Default is 0.
    dim_case_embd : int, optional
        Dimension of case-specific embedding. Default is 0.
    activation_function : str, optional
        Activation function for MLPs. Default is "tanh".
    precision : str, optional
        Numerical precision. Default uses global default.
    mixed_types : bool, optional
        If True, use a uniform fitting net for all atom types; otherwise one-per-type. Default True.
    rcond : float, optional
        Condition number for regression of atomic energy. Default None.
    seed : int | list[int], optional
        Random seed(s) for deterministic initialization. Default None.
    exclude_types : list[int], optional
        Atom types to exclude from contributions. Default [].
    type_map : list[str], optional
        Type name map for atoms. Default None.
    use_aparam_as_mask : bool, optional
        If True, `aparam` is used only as mask, not concatenated. Default False.
    default_fparam : list[float], optional
        Default frame parameter values used when dataset lacks `fparam.npy`. Default None.

    Notes
    -----
    - Inherits GeneralFitting to remain compatible with the existing architecture:
      statistics handling, masking, case embeddings, and dtype/device conventions.
    - Follows dual-MLP design inspired by EnergyFittingNetDirect: two parallel heads
      computed in a single forward pass.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int = 1,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        default_fparam: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        self.dim_out = dim_out
        super().__init__(
            var_name="energy",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            type_map=type_map,
            use_aparam_as_mask=use_aparam_as_mask,
            default_fparam=default_fparam,
            **kwargs,
        )
        # The parent builds `self.filter_layers` for energy branch.
        # Build an identical branch for latent charge `q`.
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )
        self.filter_layers_q = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self._net_out_dim(),
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii + 1024),
                    trainable=self.trainable,
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

    def _net_out_dim(self) -> int:
        """Output dimension of each branch."""
        return self.dim_out

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [self._net_out_dim()],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    "q_latent",
                    [self._net_out_dim()],
                    reducible=False,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute energy and latent charge in parallel from the same descriptor input.

        The energy branch uses GeneralFitting's standard pipeline (bias addition,
        vacuum removal, type masking). The charge branch mirrors the same input
        preparation and type-masking, producing a latent per-atom scalar `q`.
        """
        # Energy branch via the common path
        energy_common = self._forward_common(
            descriptor, atype, gr, g2, h2, fparam, aparam
        )
        energy_out = energy_common["energy"]

        # Prepare inputs for the charge branch (mirrors _forward_common)
        prec = self.prec
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape

        if self.numb_fparam > 0 and fparam is None:
            # use default fparam
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])

        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        xx_zeros = (
            torch.zeros_like(xx)
            if self.remove_vaccum_contribution is not None
            else None
        )
        net_dim_out = self._net_out_dim()

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
            )

        # check fparam dim, concate to input descriptor
        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    "get an input fparam of dim {fparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_fparam}.",
                )
            fparam = fparam.view([nf, self.numb_fparam])
            nb, _ = fparam.shape
            #t_fparam_avg = torch.tile(self.fparam_avg.view([1, self.numb_fparam]), [nb, 1])
            #t_fparam_inv_std = torch.tile(
            #    self.fparam_inv_std.view([1, self.numb_fparam]), [nb, 1]
            #)
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = torch.cat(
                [xx, fparam], 
                dim=-1,
            )
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, fparam], 
                    dim=-1,
                )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    f"get an input aparam of dim {aparam.shape[-1]}, ",
                    f"which is not consistent with {self.numb_aparam}.",
                )
            aparam = aparam.view([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            #t_aparam_avg = torch.tile(self.aparam_avg.view([1, 1, self.numb_aparam]), [nb, nloc, 1])
            #t_aparam_inv_std = torch.tile(
            #    self.aparam_inv_std.view([1, 1, self.numb_aparam]), [nb, nloc, 1]
            #)
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat(
                [xx, aparam], 
                dim=-1,
            )
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, aparam], 
                    dim=-1,
                )

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = torch.tile(self.case_embd.reshape([1, 1, -1]), [nf, nloc, 1])
            xx = torch.cat(
                [xx, case_embd], 
                dim=-1,
            )
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, case_embd], 
                    dim=-1,
                )

        outs_q = torch.zeros(
            (nf, nloc, net_dim_out),
            dtype=prec, 
            device=descriptor.device,
        ) # jit assertion
        
        if self.mixed_types:
            atom_q = self.filter_layers_q.networks[0](xx)
            if xx_zeros is not None:
                atom_q -= self.filter_layers_q.networks[0](xx_zeros)
            outs_q = outs_q + atom_q
        else:
            for type_i, ll in enumerate(self.filter_layers_q.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, net_dim_out))
                atom_q = ll(xx)
                if xx_zeros is not None:
                    atom_q -= ll(xx_zeros)
                atom_q = torch.where(mask, atom_q, 0.0)
                outs_q = outs_q + atom_q
        mask = self.emask(atype).to(torch.bool)
        outs_q = torch.where(mask[:, :, None], outs_q, 0.0)

        result = {
            "energy": energy_out.to(env.GLOBAL_PT_FLOAT_PRECISION),
            "q_latent": outs_q.to(env.GLOBAL_PT_FLOAT_PRECISION),
        }
        if "middle_output" in energy_common:
            result["middle_output"] = energy_common["middle_output"].to(
                env.GLOBAL_PT_FLOAT_PRECISION
            )
        return result
