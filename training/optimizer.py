# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import inspect
import itertools
import logging
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import hydra

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class Optimizer:
    def __init__(self, optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0, 0)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for _, set_of_schedulers in enumerate(self.schedulers):
            for option, _ in set_of_schedulers.items():
                assert option in self.optimizer.defaults, (
                    "Optimizer option "
                    f"{option} not found in {self.optimizer}. Valid options are "
                    f"{self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float, step: int) -> None:
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                if "step" in inspect.signature(scheduler.__call__).parameters:
                    new_value = scheduler(step=step, where=where)
                elif (
                    hasattr(scheduler, "scheduler")
                    and "step"
                    in inspect.signature(scheduler.scheduler.__call__).parameters
                ):
                    # To handle ValueScaler wrappers
                    new_value = scheduler(step=step, where=where)
                else:
                    new_value = scheduler(where)
                param_group[option] = new_value

    def step(self, where, step, closure=None):
        self.step_schedulers(where, step)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)


def set_default_parameters(
    scheduler_cfgs: List[DictConfig], all_parameter_names: Set[str]
) -> None:
    """Set up the "default" scheduler with the right parameters.

    Args:
        scheduler_cgfs: A list of scheduler configs, where each scheduler also
            specifies which parameters it applies to, based on the names of parameters
            or the class of the modules. At most one scheduler is allowed to skip this
            specification, which is used as a "default" specification for any remaining
            parameters.
        all_parameter_names: Names of all the parameters to consider.
    """
    constraints = [
        scheduler_cfg.parameter_names
        for scheduler_cfg in scheduler_cfgs
        if scheduler_cfg.parameter_names is not None
    ]
    if len(constraints) == 0:
        default_params = set(all_parameter_names)
    else:
        default_params = all_parameter_names - set.union(*constraints)
    default_count = 0
    for scheduler_cfg in scheduler_cfgs:
        if scheduler_cfg.parameter_names is None:
            scheduler_cfg.parameter_names = default_params
            default_count += 1
    assert default_count <= 1, "Only one scheduler per option can be default"
    if default_count == 0:
        # No default scheduler specified, add a default, but without any scheduler
        # for that option
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]], named_parameters: Dict[str, Tensor]
) -> List[torch.nn.Parameter]:
    """Return parameters which match the intersection of parameter constraints.

    Note that this returns the parameters themselves, not their names.

    Args:
        param_constraints: A list, with each element being a set of allowed parameters.
        named_parameters: Mapping from a parameter name to the parameter itself.

    Returns:
        A list containing the parameters which overlap with _each_ constraint set from
        param_constraints.
    """
    matching_names = set.intersection(*param_constraints)
    return [value for name, value in named_parameters.items() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[List[Dict]],
    named_parameters: Dict[str, Tensor],
) -> Tuple[List[Dict[Any, Any]], List[Dict[str, List[torch.nn.Parameter]]]]:
    """Produce parameter groups corresponding to all the scheduler configs.

    Takes all the scheduler configs, each of which applies to a specific optimizer
    option (like "lr" or "weight_decay") and has a set of parameter names which it
    applies to, and produces a final set of param groups where each param group
    covers all the options which apply to a particular set of parameters.

    Args:
        all_scheduler_cfgs: All the scheduler configs covering every option.
        named_parameters: Mapping from a parameter name to the parameter itself.
    Returns:
        Tuple of lists of schedulers and param_groups, where schedulers[i]
        applies to param_groups[i].
    """

    scheduler_cfgs_per_param_group = itertools.product(*all_scheduler_cfgs)
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [
            scheduler_cfg["parameter_names"] for scheduler_cfg in scheduler_cfgs
        ]
        matching_parameters = name_constraints_to_parameters(
            param_constraints, named_parameters
        )
        if len(matching_parameters) == 0:  # If no overlap of parameters, skip
            continue
        schedulers_for_group = {
            scheduler_cfg["option"]: scheduler_cfg["scheduler"]
            for scheduler_cfg in scheduler_cfgs
            if "option" in scheduler_cfg
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    """Check that the param groups are non-overlapping and cover all the parameters.

    Args:
        param_groups: List of all param groups
        model: Model to validate against. The check ensures that all the model
            parameters are part of param_groups
    """
    for pg in param_groups:
        # no param should be repeated within a group
        assert len(pg["params"]) == len(set(pg["params"]))
    parameters = [set(param_group["params"]) for param_group in param_groups]
    model_parameters = {parameter for _, parameter in model.named_parameters()}
    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Scheduler generated param_groups should be disjoint"
    assert set.union(*parameters) == model_parameters, (
        "Scheduler generated param_groups must include all parameters of the model."
        f" Found {len(set.union(*parameters))} params whereas model has"
        f" {len(model_parameters)} params"
    )


def unix_module_cls_pattern_to_parameter_names(
    filter_module_cls_names: List[str],
    module_cls_to_param_names: Dict[Type, str],
) -> Union[None, Set[str]]:
    """Returns param names which pass the filters specified in filter_module_cls_names.

    Args:
        filter_module_cls_names: A list of filter strings containing class names, like
            ["torch.nn.LayerNorm", "torch.nn.BatchNorm2d"]
        module_cls_to_param_names: Mapping from module classes to the parameter names
            they contain. See `get_module_cls_to_param_names`.
    """
    if filter_module_cls_names is None:
        return set()
    allowed_parameter_names = []
    for module_cls_name in filter_module_cls_names:
        module_cls = hydra.utils.get_class(module_cls_name)
        if module_cls not in module_cls_to_param_names:
            raise AssertionError(
                f"module_cls_name {module_cls_name} does not "
                "match any classes in the model"
            )
        matching_parameters = module_cls_to_param_names[module_cls]
        assert (
            len(matching_parameters) > 0
        ), f"module_cls_name {module_cls_name} does not contain any parameters in the model"
        logging.info(
            f"Matches for module_cls_name [{module_cls_name}]: {matching_parameters} "
        )
        allowed_parameter_names.append(matching_parameters)
    return set.union(*allowed_parameter_names)


def unix_param_pattern_to_parameter_names(
    filter_param_names: Optional[List[str]],
    parameter_names: Dict[str, torch.Tensor],
) -> Union[None, Set[str]]:
    """Returns param names which pass the filters specified in filter_param_names.

    Args:
        filter_param_names: A list of unix-style filter strings with optional
            wildcards, like ["block.2.*", "block.2.linear.weight"]
        module_cls_to_param_names: Mapping from module classes to the parameter names
            they contain. See `get_module_cls_to_param_names`.
    """

    if filter_param_names is None:
        return set()
    allowed_parameter_names = []
    for param_name in filter_param_names:
        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        assert (
            len(matching_parameters) >= 1
        ), f"param_name {param_name} does not match any parameters in the model"
        logging.info(f"Matches for param_name [{param_name}]: {matching_parameters}")
        allowed_parameter_names.append(matching_parameters)
    return set.union(*allowed_parameter_names)


def _unix_pattern_to_parameter_names(
    scheduler_cfg: DictConfig,
    parameter_names: Set[str],
    module_cls_to_param_names: Dict[Type, str],
) -> Union[None, Set[str]]:
    """Returns param names which pass the filters specified in scheduler_cfg.

    Args:
        scheduler_cfg: The config for the scheduler
        parameter_names: The set of all parameter names which will be filtered
    """
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    return unix_param_pattern_to_parameter_names(
        scheduler_cfg.get("param_names"), parameter_names
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            scheduler_cfg.get("module_cls_names"), module_cls_to_param_names
        )
    )


def get_module_cls_to_param_names(
    model: nn.Module, param_allowlist: Set[str] = None
) -> Dict[Type, str]:
    """Produce a mapping from all the modules classes to the names of parames they own.

    Only counts a parameter as part of the immediate parent module, i.e. recursive
    parents do not count.

    Args:
        model: Model to iterate over
        param_allowlist: If specified, only these param names will be processed
    """

    module_cls_to_params = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())
        for param_name, _ in module.named_parameters(recurse=False):
            full_param_name = get_full_parameter_name(module_name, param_name)
            if param_allowlist is None or full_param_name in param_allowlist:
                module_cls_to_params[module_cls].add(full_param_name)
    return module_cls_to_params


def construct_optimizer(
    model: torch.nn.Module,
    optimizer_conf: Any,
    options_conf: Mapping[str, List] = None,
    param_group_modifiers_conf: List[Callable] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups=True,
) -> Optimizer:
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888

    Args:
        model: model to perform stochastic gradient descent
            optimization or ADAM optimization.
        optimizer_conf: Hydra config consisting a partial torch optimizer like SGD or
            ADAM, still missing the params argument which this function provides to
            produce the final optimizer
        param_group_modifiers_conf: Optional user specified functions which can modify
            the final scheduler configs before the optimizer's param groups are built
        param_allowlist: The parameters to optimize. Parameters which are not part of
            this allowlist will be skipped.
        validate_param_groups: If enabled, valides that the produced param_groups don't
            overlap and cover all the model parameters.
    """
    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}

    named_parameters = {
        name: param
        for name, param in model.named_parameters()
        if name in param_allowlist
    }

    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return Optimizer(optimizer)

    all_parameter_names = {
        name for name, _ in model.named_parameters() if name in param_allowlist
    }
    module_cls_to_all_param_names = get_module_cls_to_param_names(
        model, param_allowlist
    )

    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config.option = option
            config.parameter_names = _unix_pattern_to_parameter_names(
                config, all_parameter_names, module_cls_to_all_param_names
            )
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)

    if param_group_modifiers_conf:
        for custom_param_modifier in param_group_modifiers_conf:
            custom_param_modifier = hydra.utils.instantiate(custom_param_modifier)
            all_scheduler_cfgs = custom_param_modifier(
                scheduler_cfgs=all_scheduler_cfgs, model=model
            )
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )
    if validate_param_groups:
        validate_param_group_params(param_groups, model)
    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return Optimizer(optimizer, schedulers)


def get_full_parameter_name(module_name, param_name):
    if module_name == "":
        return param_name
    return f"{module_name}.{param_name}"


class GradientClipper:
    """
    Gradient clipping utils that works for DDP
    """

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        assert isinstance(max_norm, (int, float)) or max_norm is None
        self.max_norm = max_norm if max_norm is None else float(max_norm)
        self.norm_type = norm_type

    def __call__(self, model: nn.Module):
        if self.max_norm is None:
            return  # no-op

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
        )


class ValueScaler:
    def __init__(self, scheduler, mult_val: float):
        self.scheduler = scheduler
        self.mult_val = mult_val

    def __call__(self, *args, **kwargs):
        val = self.scheduler(*args, **kwargs)
        return val * self.mult_val


def rgetattr(obj, rattrs: str = None):
    """
    Like getattr(), but supports dotted notation for nested objects.
    rattrs is a str of form 'attr1.attr2', returns obj.attr1.attr2
    """
    if rattrs is None:
        return obj
    attrs = rattrs.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict]],
    model,
    layer_decay_value: float,
    layer_decay_min: Optional[float] = None,
    apply_to: Optional[str] = None,
    overrides: List[Dict] = (),
) -> List[List[Dict]]:
    """
    Args
    - scheduler_cfgs: a list of omegaconf.ListConfigs.
        Each element in the list is a omegaconfg.DictConfig with the following structure
        {
            "scheduler": <some fvcore scheduler>
            "option": <value> possible options are "lr", "weight_decay" etc.
            "parameter_names": Set of str indicating param names that this scheduler applies to
        }
    - model: a model that implements a method `get_layer_id` that maps layer_name to an integer and
            and a method get_num_layers.
            Alternatively, use apply_to argument to select a specific component of the model.
    - layer_decay_value: float
    - layer_decay_min: min val for layer decay
    - apply_to: optional arg to select which component of the model to apply the the layer decay modifier to
    - overrides: to manually override lr for specific patterns. Is a list of dicts. Each dict, has keys "pattern", "value".
    Returns
    - scheduler_configs: same structure as the input, elements can be modified
    """
    model = rgetattr(model, apply_to)
    num_layers = model.get_num_layers() + 1
    layer_decays = [
        layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)
    ]
    if layer_decay_min is not None:
        layer_decays = [max(val, layer_decay_min) for val in layer_decays]
    final_scheduler_cfgs = []
    # scheduler_cfgs is a list of lists
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group = []
        # scheduler_cfg_group is a list of dictionaries
        for scheduler_cfg in scheduler_cfg_group:
            if scheduler_cfg["option"] != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            # Need sorted so that the list of parameter names is deterministic and consistent
            # across re-runs of this job. Else it was causing issues with loading the optimizer
            # state during a job restart (D38591759)
            parameter_names = sorted(scheduler_cfg["parameter_names"])

            # Only want one cfg group per layer
            layer_cfg_groups = {}
            for param_name in parameter_names:
                layer_id = num_layers
                this_scale = layer_decays[layer_id]
                if param_name.startswith(apply_to):
                    layer_id = model.get_layer_id(param_name)
                    this_scale = layer_decays[layer_id]
                    # Overrides
                    for override in overrides:
                        if fnmatch.fnmatchcase(param_name, override["pattern"]):
                            this_scale = float(override["value"])
                            layer_id = override["pattern"]
                            break

                if layer_id not in layer_cfg_groups:
                    curr_param = {
                        "option": scheduler_cfg["option"],
                        "scheduler": ValueScaler(
                            scheduler_cfg["scheduler"], this_scale
                        ),
                        "parameter_names": {param_name},
                    }
                else:
                    curr_param = layer_cfg_groups[layer_id]
                    curr_param["parameter_names"].add(param_name)
                layer_cfg_groups[layer_id] = curr_param

            for layer_cfg in layer_cfg_groups.values():
                curr_cfg_group.append(layer_cfg)

        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs
