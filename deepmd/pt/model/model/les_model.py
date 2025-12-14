# SPDX-License-Identifier: LGPL-3.0-or-later
from traceback import print_tb
from typing import Any, Optional, Dict
import logging

import torch

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.utils import (
    env,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
    fit_output_to_model_output,
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

log = logging.getLogger(__name__)

LesModel_ = make_model(DPAtomicModel)


@BaseModel.register("les")
class LesModel(DPModelCommon, LesModel_):
    model_type = "les"

    def __init__(
        self,
        *args: Any,
        les_config: Optional[str] = None,
        les_arguments: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        LesModel_.__init__(self, *args, **kwargs)
        try:
            from les import Les
            if les_arguments is None:
                if les_config is not None:
                    import yaml
                    with open(les_config, "r") as f:
                        les_arguments = yaml.safe_load(f) or {}
                else:
                    les_arguments = {"use_atomwise": False}
                    
            log.info(f"Les arguments: {les_arguments}")
            self._les = Les(les_arguments=les_arguments)
        except ImportError:
            raise ImportError("LesModel requires LES module. See `https://github.com/ChengUCB/les`")

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
            "q_latent": out_def_data["q_latent"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward_common(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        cc, bb, fp, ap, input_prec = self.input_type_cast(
            coord, box=box, fparam=fparam, aparam=aparam
        )
        del coord, box, fparam, aparam
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            cc,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=bb,
        )
        model_predict_lower = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            do_atomic_virial=do_atomic_virial,
            fparam=fp,
            aparam=ap,
            cell=bb,
        )
        model_predict = communicate_extended_output(
            model_predict_lower,
            self.model_output_def(),
            mapping,
            do_atomic_virial=do_atomic_virial,
        )

        # Correct the virial
        if "energy_derv_c_redu" in model_predict:
            cell_les = getattr(self, "_last_cell_les", None)
            virial_corr = self.correct_virial(model_predict["energy_redu"], cell_les)
            if virial_corr is not None:
                model_predict["energy_derv_c_redu"] = (
                    model_predict["energy_derv_c_redu"] + virial_corr.to(model_predict["energy_derv_c_redu"].dtype)
                )

        model_predict = self.output_type_cast(model_predict, input_prec)
        return model_predict

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        extra_nlist_sort: bool = False,
        cell: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        nframes, nall = extended_atype.shape[:2]
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlist = self.format_nlist(
            extended_coord, extended_atype, nlist, extra_nlist_sort=extra_nlist_sort
        )
        cc_ext, box_cast, fp, ap, input_prec = self.input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        del extended_coord, fparam, aparam
        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
            comm_dict=comm_dict,
        )
        
        # use `q_latent` to calculate the Ewald summation
        if "q_latent" in atomic_ret:
            nf, nloc, qdim = atomic_ret["q_latent"].shape
            coord_loc = cc_ext[:, :nloc, :]
            if cell is not None:
                cell_les = cell.view(nf, 3, 3).to(device=coord_loc.device, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
                cell_les.requires_grad_(True)
            else:
                # if cell is not provided, set it to zero; which means the non-pbc case
                cell_les = torch.zeros((nf, 3, 3), device=coord_loc.device, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
                cell_les.requires_grad_(False)

            self._last_cell_les = cell_les # for correct the virial
            q = atomic_ret["q_latent"].squeeze(-1)
            q_flat = q.reshape(-1)
            r_flat = coord_loc.reshape(-1, 3)
            batch = torch.arange(nf, device=r_flat.device).repeat_interleave(nloc)
            les_result = self._les(
                latent_charges=q_flat,
                positions=r_flat,
                cell=cell_les,
                batch=batch,
                compute_energy=True,
            )

            # update the energy
            e_lr = les_result.get("E_lr", None)
            if e_lr is not None:
                pot_per_atom = (e_lr.view(nf, 1).to(atomic_ret["energy"].dtype) / nloc).view(nf, 1, 1).expand(nf, nloc, 1)
                atomic_ret["energy"] = atomic_ret["energy"] + pot_per_atom

        model_predict = fit_output_to_model_output(
            atomic_ret,
            self.atomic_output_def(),
            cc_ext,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=atomic_ret["mask"] if "mask" in atomic_ret else None,
        )
        model_predict = self.output_type_cast(model_predict, input_prec)
        return model_predict

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if "q_latent" in model_ret:
            model_predict["q_latent"] = model_ret["q_latent"]
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
        
    def correct_virial(self, energy, cell):
        if cell is None:
            return None
        if not cell.requires_grad:
            return None
        faked_grad = torch.ones_like(energy)
        lst = torch.jit.annotate(list[Optional[torch.Tensor]], [faked_grad])
        dE_dh = torch.autograd.grad(
            [energy],
            [cell],
            grad_outputs=lst,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        assert dE_dh is not None
        virial_corr_mat = -torch.einsum("bji,bjk->bik", dE_dh, cell)
        nf = virial_corr_mat.shape[0]
        virial_corr = virial_corr_mat.view(nf, 1, 9)
        return virial_corr

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            cell=cell,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if "q_latent" in model_ret:
            model_predict["q_latent"] = model_ret["q_latent"]
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        return model_predict
